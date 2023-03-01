import numpy as np
import torch
def calc_jacobian( inp,model):


    ############## prepare the static model
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    ############## find Jacobian
    if inp.requires_grad is False: 
        inp.requires_grad =True
    out = model(inp)
    jacob = []
    
    # start_time_total = time.time()
    for i in range(inp.size()[2]):
        for j in range(inp.size()[3]):
            part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True) 
            jacob.append( part_der[0][0,0].data.view(-1)) 
    #print("----total time to compute jacobian --- %s seconds ---" % (time.time() - start_time_total))

    return torch.stack(jacob)




def calc_jacobian_rows( inp,model, i,j):
    '''
    @ im: torch image C,H,W
    '''
    ############## prepare the static model
    for param in model.parameters():
        param.requires_grad = False


    model.eval()

    if inp.requires_grad is False: 
        inp.requires_grad =True
    ##############  Jacobian
    out = model(inp)

    part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True)
    jacob =  part_der[0][0,0] 

    
    return jacob

def calc_jacobian_row_MS( inp,models, i,j, device):
    '''
    @ im: torch image C,H,W
    '''
    ############## prepare the static model
    for k,model in  models.items():
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        
    if inp.requires_grad is False: 
        inp.requires_grad =True
        
    ##############  Jacobian
    out = multi_scale_denoising(inp, models, device)

    part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True) 
    jacob =  part_der[0][0,0] 

    
    return jacob
