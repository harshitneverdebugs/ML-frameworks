def apply_kernel(image, kernel):
    ri, ci = image.shape       
    rk, ck = kernel.shape    
    ro, co = ri-rk+1, ci-ck+1  
    output = torch.zeros([ro, co])
    for i in range(ro):
        for j in range(co):
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)
    return output