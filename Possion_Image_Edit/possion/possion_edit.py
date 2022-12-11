import numpy as np
import pdb

class PossionEdit():
    def __init__(self, src, mask, dst, pos) -> None:
        '''
        src represents g, dst represents S, pos is index of pos in dst where the src should be
        '''
        self.src = src
        h_src, w_src, _ = src.shape
        self.mask = self.get_mask(h_src, w_src) if mask is None else mask
        # pdb.set_trace()
        self.h_mask, self.w_mask, _ = self.mask.shape

        self.dst = dst
        self.pos = pos
        self.h_dst, self.w_dst, _ = dst.shape
        self.origin = self.get_origin_val()
    
    def get_mask(self, h_src, w_src):
        mask = np.full(shape=(h_src, w_src, 3), fill_value=255, dtype=np.uint8)
        mask[0, :, :] = mask[h_src-1, :, :] = mask[:, 0, :] = mask[:, w_src-1, :] = 0
        return mask


    def __call__(self):
        '''
        - mark the main part with numbers using masks?
        - construct matrix A and vector b
        - solve Ax = b
        - copy x to the dst matrix
        must consider the border!!
        '''
        self.mark_main_part()
        A, b = self.construct_A_b()
        x = self.solve_equation(A, b)
        img_edit = self.get_final_img(x)
        return img_edit
    
    def construct_A_b(self):
        '''
        construct matrix A and vector b: A is laplace operator, b is the divergence
        A should be square matrix
        '''
        mat_A = np.zeros([self.num_g, self.num_g, 3], np.int32) # (size(g), size(g), 3)
        vec_b = np.zeros([self.num_g, 3]) # (size(g), 3)
        origin = self.get_origin_val()

        for h in range(0, self.h_mask):
            for w in range(0, self.w_mask):
                # judge it's in g or not
                if (self.mask[h, w] == 255).all():
                    # judge it's in main part or not 
                    b = self.get_div_mixed_grad(h, w) # if in main part, it is div

                    for neighbor in self.neighbors(h, w):
                        # 对于每一个neighbor
                        # if self.core_table[neighbor[0], neighbor[1]]:
                        if (self.mask[neighbor] == 255).all():
                            mat_A[self.index_to_num_g[h, w], self.index_to_num_g[neighbor]] = \
                                np.array([1, 1, 1])
                        else:
                            b -= origin[neighbor] #! 当neighbor不是中心区域的时候，就不会给它赋值1，仍然为0，并且b也不是div，要减掉对应的原图的值？
                        
                    mat_A[self.index_to_num_g[h, w], self.index_to_num_g[h, w]] = np.array([-4, -4, -4]) # if in main part, laplace operator
                    vec_b[self.index_to_num_g[h, w]] = b
        # np.savetxt("./mat_A", mat_A[:, :, 0])
        return mat_A, vec_b

    def get_origin_val(self):
        '''
        get the originial pixel values in the dst
        '''
        _back = np.zeros([self.h_mask, self.w_mask, 3])
        for h in range(self.h_mask):
            for w in range(self.w_mask):
                _back[h, w] = self.dst[h + self.pos[0], w + self.pos[1]] 
        return _back

    def solve_equation(self, mat_A, vec_b):
        '''
        solve the equation: Ax = b
        '''
        channels = []
        for c in range(3):
            x_ans = np.linalg.solve(mat_A[:, :, c], vec_b[:, c]) # solve Ax = b (size(g), 1)
            # clamp between[0, 255]
            x_ans[x_ans < 0] = 0
            x_ans[x_ans > 255] = 255
            i = 0
            one_channel = np.zeros([self.h_mask, self.w_mask])
            # 将解出来的g的像素按照g的形状排列一下
            for h in range(self.h_mask):
                for w in range(self.w_mask):
                    if (self.mask[h, w] == 255).all(): # if g
                        one_channel[h, w] = x_ans[i]
                        i += 1

            channels.append(one_channel)
        x_ans = np.dstack(channels).astype(np.uint8) # 3 x (size(g), size(g)) => (size(g), size(g), 3)
        return x_ans

    def get_final_img(self, x):
        '''
        copy the solved x to the dst img, x: (h_mask, w_mask)
        '''
        img_edit = self.dst.copy()
        for h in range(self.h_mask):
            for w in range(self.w_mask):
                if (self.mask[h, w] == 255).all():
                    i, j = h + self.pos[0], w + self.pos[1]
                    img_edit[i, j] = x[h, w]
        return img_edit

    def get_div(self, h, w):
        '''
        get the div of the src[h,w]
        '''
        return sum([self.src[neighbor] - self.src[h, w] for neighbor in self.neighbors(h, w)])
    
    def get_div_mixed_grad(self, h, w):
        div = 0
        for n in self.neighbors(h, w):
            # if the src's grad is larger
            if abs(self.src[n] - self.src[h, w]).sum() > abs(self.dst[n] - self.dst[h, w]).sum(): 
                div += self.src[n] - self.src[h, w]
            else:
                div += self.dst[n] - self.dst[h, w]
        return div


    def neighbors(self, h, w):
        return [(h + 1, w), (h - 1, w), (h, w + 1), (h, w - 1)]

    def mark_main_part(self):
        '''
        count the number of pixels in g except the border
        '''
        self.num_g = 0
        self.index_to_num_g = np.zeros([self.h_mask, self.w_mask], np.uint16) # 2d index => number of obk
        for h in range(0, self.h_mask):
            for w in range(0, self.w_mask):
               
                if (self.mask[h, w] == 255).all(): # 只要255的白色部分
                    self.index_to_num_g[h, w] = self.num_g
                    self.num_g += 1

    def is_border(self, h, w):
        '''
        judge the pixel in the mask is border or not
        yes: true, no: false
        '''
        for n in self.neighbors(h, w):
            if self.is_pos_valid(*n):
                # 当附近的节点都在合理范围内的时候，就考察这个邻居节点是否为mask为0的部分
                if not self.mask[n].all(): # on the border
                    return True
            else:
                # 当附近的节点不在合理范围内，那说明这个点就在mask的boder上
                return True
        return False
    
    def is_pos_valid(self, h, w):
        """
        judge whether the index of pos is in the range or not
        True: in the range False: not 
        """
        if h >= 0 and h < self.h_mask and w >= 0 and w < self.w_mask:
            return True
        return False

                            
    
    


        
