from scipy.io import loadmat

import sph_util

def matToPly(plyName,matName):#把mat中的位置信息转换到ply中
    pos = loadmat(matName)['pos']
    T = pos.shape[0]
    N=  pos.shape[1]
    print(pos.shape)

    for t in range(1,T+1):

        sph_util.write_ply(plyName, t, 3, N, pos[t-1],[], needVel=False)
def velpredToPly(velMat,plyName):#根据vel生成粒子路径，并生成ply
    vel = loadmat(velMat)['vel_pred']# T * N * 3
    pos = loadmat('pred_vel/pos.mat')['pos']# T * N * 3 需要知道粒子的初始位置

    T=vel.shape[0]
    N=vel.shape[1]
    pos2=[]
    pos2.append(pos[0])
    deltaTime=0.1012
    print(pos2)
    for x in range(0,T):
        pos2.append(pos2[x]+vel[x]*deltaTime)

    for t in range(1,T+1):
        sph_util.write_ply(plyName, t, 3, N, pos2[t-1],[], needVel=False)


# velToPos('','../data/vel_pred.mat')
# matToPly('./output-230228/solidCut','./output-230228/solidCut.mat')
# matToPly('./output-230228/solidPos','./output-230228/solidPos.mat')
# matToPly('./plyFromMat/groundTruth/plyData_Frame','plyOutput-14/pos.mat')
# velpredToPly('plyOutput-14/vel_pred.mat','./plyFromMat/prediction/plyData_Frame')#根据vel生成粒子路径
velpredToPly('pred_vel/vel_pred_bound_soft3.mat', 'pred_vel/ply_12/frame')