import numpy as np
import cv2 as cv
# from skimage.metrics import structural_similarity as ssim
def patch_mae(p1, p2):
    #TODO: Ensure returns a numerical value
    diff = p1.astype(np.float32) - p2.astype(np.float32)
    return np.mean(np.abs(diff))

# def patch_sae(p1, p2):
#     diff = p1.astype(np.float32) - p2.astype(np.float32)
#     return np.sum(np.abs(diff))

class ViewTracker:
    def __init__(self,base_image,verbose=False):
        self.verbose = verbose
        self.set_image(base_image)
        self.data = None

    def clear_map(self):
        self.view_map  = np.zeros((self.h,self.w),dtype=np.uint8)
        self.view_recon  = np.zeros((self.h,self.w,3),dtype=np.uint8)
        self.views = []
        self.data = None
        if self.verbose: print("Map/views Cleared.")

    def set_image(self,image,clear=True):
        self.base_image = image
        self.h, self.w, _ = self.base_image.shape
        if clear:
            self.clear_map()

    def add_view(self,image,focus):
        # Focus properties
        f_pos = focus.pos
        f_size = focus.size
        f_k = focus.k_size
        # Position, size, area, normed wrt image
        rel_pos = [f_pos[0]/self.w, f_pos[1]/self.h]
        rel_size = [f_size/self.w, f_size/self.h]
        rel_area = (f_size * f_size) / (self.w * self.h)
        # Top left and bottom right bounding box coords of focus on image
        xy1 = [int(f_pos[0] - (f_size/2)), int(f_pos[1] - (f_size/2))]
        xy2 = [int(f_pos[0] + (f_size/2)), int(f_pos[1] + (f_size/2))]
        
        # Update view map
        roi = self.view_map[xy1[1]:xy2[1],xy1[0]:xy2[0]]
        value = f_k # Raw viewmap values = k values
        mask = (roi == 0) | (roi > value) # The area which is unseen or seen at higher k
        roi[mask] = value
        # Update reconstruction
        recon_roi = self.view_recon[xy1[1]:xy2[1],xy1[0]:xy2[0]]
        mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # recon_mask = np.zeros_like(self.view_recon, dtype=np.uint8)
        # self.view_recon[xy1[1]:xy2[1],xy1[0]:xy2[0]] = mask
        recon_img = focus.reconstruct()
        recon_img = cv.resize(recon_img,(xy2[1]-xy1[1],xy2[0]-xy1[0]))
        recon_img[~mask] = 0
        # nonzero_mask = np.any(recon_img != 0, axis=2)
        print(f"S1: {recon_img[mask].shape}, S2: { recon_roi[mask].shape}")
        mae = patch_mae(recon_img[mask], recon_roi[mask])
        # sae = patch_sae(recon_img[mask], recon_roi[mask])

        recon_roi[mask] = recon_img[mask]
        self.view_recon[xy1[1]:xy2[1],xy1[0]:xy2[0]] = recon_roi
        # cv.imshow("recon debug",recon_img)
        # cv.imshow("recon debu2g",mask)
        # cv.imshow("recon debu23g",recon_roi)

        # Calculate scoring
        # TODO: Compare the difference between the recon before and after update, this will be an important score
        # penalty = 0.2
        # max_score = (f_size * f_size) / (1 + (penalty * (f_k-1) * (f_k-1)))
        # TODO: There's probably a way to save compute if I do this earlier
        mask_contrib = np.sum(mask) / (f_size * f_size) # [0-1] how much of the focus changed the current map
        view_score = mask_contrib * mae
        # k_s = f_k / f_size
        # Assemble features & data
        if mask_contrib > 0.0:
            features = [rel_pos[0], rel_pos[1], rel_size[0], rel_size[1], rel_area, f_k]
            data = focus.get_data()
            if self.data is None:
                self.data = data
            else:
                self.data = np.vstack((self.data,data))
            self.views.append((features,data,view_score))
        
            # Debug print
            if self.verbose:
                print(f"[Add View]\n  [Image] {image.shape}")
                print(f"  [Focus]\n\tRelative Pos: {rel_pos}\n\tRelative Size: {rel_size}\n\tK Value: {f_k}")
                print(f"  [Features]\n\t{features}")
                print(f"  [Scores]\n\tMask: {mask_contrib}")
                print(f"\tMAE: {mae}\n\tView Score: {view_score:.2f}")

        else:
            print("WARN: Nothing new added, view not appended.")

        

    def show_map(self, wait=1, close_win=False, show_raw=False, show_recon=True, show_data=True):
        # Determine max value for scaling
        max_val = max(10,self.view_map.max()) 
        if self.verbose: print(f"Max: {max_val}")

        # Create a scaled/inverted display map
        self.disp_image = np.zeros_like(self.view_map, dtype=np.uint8)
        self.disp_image = self.view_map * (255 // (max_val + 2)) # Scale map to fit within 255 nicely
        z_mask = self.disp_image == 0   # Create a mask for all zero values
        self.disp_image[z_mask] = 255   # Set all zero values to 255
        self.disp_image = 255 - self.disp_image # Invert map 
        self.disp_image = cv.resize(self.disp_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        # Show display map
        cv.imshow("Disp Map", self.disp_image)

        if show_recon:
            self.view_recon
            self.disp_recon = self.view_recon.copy()
            self.disp_recon = cv.resize(self.disp_recon, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
            cv.imshow("Recon Map", self.disp_recon)

        # Show Raw View Map ? 
        if show_raw:
            self.disp_image2 = self.view_map.copy()
            self.disp_image2 = cv.resize(self.disp_image2, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
            cv.imshow("View Map", self.disp_image2)

        if show_data:
            self.show_data()

        cv.waitKey(wait)
        if close_win: cv.destroyWindow("Disp Map")

    def show_data(self,h=10):
        if self.data is not None:
            data_disp = np.repeat(self.data, h, axis=0)
            data_disp = cv.resize(data_disp, (self.disp_image.shape[1],data_disp.shape[0]), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
           
            cv.imshow("View Tracker Data", data_disp)

    def print_states(self):
        print(f"View Tracker States: ")
        score_sum = 0
        if len(self.views) > 0:
            print(f"[{' Pos':<15}][{' Size':<15}][{' Area':<15}][{' K':<15}][{' Data Shape':<15}][{' Score':<15}]")
            for f, data, score in self.views:
                score_sum += score
                s =  f"   {f[0]:.2f}, {f[1]:.2f}".ljust(17)
                s += f"   {f[2]:.2f}, {f[3]:.2f}".ljust(17)
                s += f"   {f[4]:.4f}".ljust(17)
                s += f"   {f[5]}".ljust(17)
                
                shape = data.shape
                s += f"   {data.shape}".ljust(17)
                s += f"   {score:.2f}".ljust(17)
                print(s)
            print(f"Total Views: {len(self.views)}  -  Total Score: {score_sum:.2f}")
            total_features = np.prod(shape) * len(self.views)
            print(f"Total Features (Views x 6 + Shape x Channels): {total_features}")



