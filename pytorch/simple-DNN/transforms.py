
class MinMaxNormalize():
    def __call__(self, sample):
        sample_shape  = sample.size()
        flattened_img = sample.flatten()
        norm_img      = (flattened_img - flattened_img.min()) / (flattened_img.max() - flattened_img.min())
        ready_img     = norm_img.reshape(sample_shape)
        return ready_img

class Flatten():
    def __call__(self, sample):
        return sample.flatten()
