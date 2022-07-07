import torch
import torch.nn as nn
import torch.nn.functional as F 

# A pytorch implement of CCL as described in [1]
#[1] Nie et al. Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation. TCSVT, 2021.

############ usage ############
# feature_1: bs, c, h, w      --- feature maps encoded from img1
# feature_2: bs, c, h, w      --- feature maps encoded from img2
# correlation: bs, 2, h, w    --- the correlation flow
correlation = CCL(feature_1, feature_2)
###############################

def extract_patches(x, kernel=3, stride=1):
    if kernel != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches

    
def CCL(feature_1, feature_2):
    bs, c, h, w = feature_1.size()
    
    norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
    norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
    #print(norm_feature_2.size())
    
    patches = extract_patches(norm_feature_2)
    if torch.cuda.is_available():
        patches = patches.cuda()
    matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))
    
    match_vol = []
    for i in range(bs):
        single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
        match_vol.append(single_match)

    match_vol = torch.cat(match_vol, 0)
    #print(match_vol .size())
    
    # scale softmax
    softmax_scale = 10
    match_vol = F.softmax(match_vol*softmax_scale,1)
    
    channel = match_vol.size()[1]
    
    h_one = torch.linspace(0, h-1, h)
    one1w = torch.ones(1, w)
    if torch.cuda.is_available():
        h_one = h_one.cuda()
        one1w = one1w.cuda()
    h_one = torch.matmul(h_one.unsqueeze(1), one1w)
    h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)
    w_one = torch.linspace(0, w-1, w)
    oneh1 = torch.ones(h, 1)
    if torch.cuda.is_available():
        w_one = w_one.cuda()
        oneh1 = oneh1.cuda()
    w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
    w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)
    
    c_one = torch.linspace(0, channel-1, channel)
    if torch.cuda.is_available():
        c_one = c_one.cuda()
    c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)
    
    flow_h = match_vol*(c_one//w - h_one)
    flow_h = torch.sum(flow_h, dim=1, keepdim=True)
    flow_w = match_vol*(c_one%w - w_one)
    flow_w = torch.sum(flow_w, dim=1, keepdim=True)
    
    feature_flow = torch.cat([flow_w, flow_h], 1)
    #print(flow.size())
    
    return feature_flow