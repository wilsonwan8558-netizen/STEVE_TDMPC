"""G9 E0/E1/E2 current-state estimators; E2 is nonlinear tracking UKF + P mapper."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
RL=HERE.parents[1]
OUT=RL/"logs/generic_solver_phase_g9_temporal_anticipatory_safety"
G3=RL/"logs/generic_solver_phase_g3_force_predictor"
DEVICE=torch.device("cuda:0")
SEED=260209


def json_new(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def csv_new(path,rows):
    import csv
    with path.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def data():
    with np.load(OUT/"development_temporal_behavior.npz") as z:
        return {k:z[k].copy() for k in z.files}


def split_mask(d,group):
    split=json.loads((OUT/"dataset_split.json").read_text())
    return np.isin(d["seed"],split[group])


def cuda_gate():
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("G9 model training requires CUDA_VISIBLE_DEVICES=0; no CPU fallback")


class Transition(nn.Module):
    """Nonlinear correction to one-control-interval constant velocity tracking."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(22,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,20))

    def forward(self,x,a):
        residual=0.2*torch.tanh(self.net(torch.cat((x,a),dim=-1)))
        p=torch.clamp(x[:,:10]+x[:,10:20]+residual[:,:10],-1,1)
        v=torch.clamp(x[:,10:20]+residual[:,10:20],-2,2)
        return torch.cat((p,v),dim=-1)


class Mapper(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,38))

    def forward(self,x):return self.net(x)


def transition_data(d):
    obs=d["history_observation"]
    p0,p1,p2=obs[:,0,:10],obs[:,1,:10],obs[:,2,:10]
    x=np.concatenate((p1,p1-p0),axis=1).astype(np.float32)
    y=np.concatenate((p2,p2-p1),axis=1).astype(np.float32)
    a=d["history_action"][:,1].astype(np.float32)
    return x,a,y


def train_transition():
    cuda_gate();d=data();x,a,y=transition_data(d)
    tr,va=split_mask(d,"train"),split_mask(d,"validation")
    torch.manual_seed(SEED);net=Transition().to(DEVICE)
    tx=torch.as_tensor(x[tr],device=DEVICE);ta=torch.as_tensor(a[tr],device=DEVICE)
    ty=torch.as_tensor(y[tr],device=DEVICE)
    vx=torch.as_tensor(x[va],device=DEVICE);v_a=torch.as_tensor(a[va],device=DEVICE)
    vy=torch.as_tensor(y[va],device=DEVICE)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    generator=torch.Generator(device="cpu").manual_seed(SEED)
    best=math.inf;patience=0;history=[]
    path=OUT/"models/transition.pt";path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    for epoch in range(60):
        net.train()
        order=torch.randperm(len(tx),generator=generator).to(DEVICE)
        for chunk in order.split(512):
            loss=(net(tx[chunk],ta[chunk])-ty[chunk]).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        net.eval()
        with torch.inference_mode():val=float((net(vx,v_a)-vy).square().mean().cpu())
        history.append({"epoch":epoch,"validation_MSE":val})
        if val<best-1e-7:
            best=val;patience=0;torch.save(net.cpu().state_dict(),path);net.to(DEVICE)
        else:
            patience+=1
            if patience>=8:break
    json_new(OUT/"transition_training.json",{"status":"VALIDATION_SELECTED","best_validation_MSE":best,
        "history":history,"training_seed":SEED,"input":"previous corrected 2D tracking positions/velocity and previous action",
        "target":"next 2D tracking positions and finite-difference velocity",
        "nonlinear_process":"constant velocity plus action/state conditioned tanh MLP correction; clip physical normalized positions",
        "no_privileged_or_future_input":True})
    print(f"G9 nonlinear tracking transition validation MSE {best:.6g}",flush=True)


def transition_numpy(net,sigma,action):
    with torch.inference_mode():
        output=net(torch.as_tensor(sigma,dtype=torch.float32),
                   torch.as_tensor(np.broadcast_to(action,(len(sigma),2)).copy(),dtype=torch.float32)).numpy()
    return np.asarray(output,dtype=np.float64)


def sigma_points(mean,cov):
    n=len(mean)
    for jitter in (1e-10,1e-8,1e-6):
        try:chol=np.linalg.cholesky(n*(cov+jitter*np.eye(n)));break
        except np.linalg.LinAlgError:continue
    else:raise RuntimeError("UKF covariance lost positive definiteness")
    return np.vstack((mean,mean+chol.T,mean-chol.T))


def ukf_episode(net,observations,actions,Q,R,noise=None):
    """2n+1 sigma-point UKF: nonlinear prediction, current measurement correction."""
    n=20;Wm=np.r_[0.,np.full(2*n,1/(2*n))];Wc=Wm.copy();Wc[0]=2.
    mean=np.r_[observations[0,:10],np.zeros(10)]
    cov=np.diag(np.r_[np.full(10,R),np.full(10,0.04)])
    features=[];diagnostics=[]
    for d,obs in enumerate(observations):
        measurement=obs[:10].astype(np.float64).copy()
        if noise is not None:
            measurement+=noise[d]
        if d:
            propagated=transition_numpy(net,sigma_points(mean,cov),actions[d-1])
            prior=np.sum(Wm[:,None]*propagated,axis=0)
            delta=propagated-prior
            prior_cov=np.einsum("i,ij,ik->jk",Wc,delta,delta)+np.diag(Q)
            z=propagated[:,:10]
            z_mean=np.sum(Wm[:,None]*z,axis=0)
            z_delta=z-z_mean
            S=np.einsum("i,ij,ik->jk",Wc,z_delta,z_delta)+R*np.eye(10)
            cross=np.einsum("i,ij,ik->jk",Wc,delta,z_delta)
            gain=np.linalg.solve(S,cross.T).T
            innovation=measurement-z_mean
            mean=prior+gain@innovation
            cov=prior_cov-gain@S@gain.T
            cov=(cov+cov.T)/2
            values,vectors=np.linalg.eigh(cov)
            cov=(vectors*np.maximum(values,1e-9))@vectors.T
            diagnostics.append((float(np.linalg.norm(innovation)),float(np.min(values)),float(np.max(np.diag(cov)))))
        features.append(np.r_[obs,mean[10:20],np.log(np.maximum(np.diag(cov)[10:20],1e-9))].astype(np.float32))
    return np.asarray(features),diagnostics


def filter_all():
    cuda_gate();d=data();model=Transition()
    model.load_state_dict(torch.load(OUT/"models/transition.pt",map_location="cpu",weights_only=True));model.eval()
    x,a,y=transition_data(d);tr=split_mask(d,"train")
    with torch.inference_mode():pred=model(torch.as_tensor(x[tr]),torch.as_tensor(a[tr])).numpy()
    residual=y[tr]-pred
    Q=np.maximum(np.var(residual,axis=0,dtype=np.float64),1e-5)
    R=1e-4
    with np.load(G3/"behavior_features.npz") as z:
        seed=z["seed"].copy();decision=z["decision_index"].copy();obs=z["observation"].copy();act=z["action"].copy()
    lookup={(int(s),int(k)):i for i,(s,k) in enumerate(zip(seed,decision))}
    output=np.empty((len(d["seed"]),34),dtype=np.float32)
    row_lookup={(int(s),int(k)):i for i,(s,k) in enumerate(zip(d["seed"],d["decision_index"]))}
    diag=[]
    for episode_seed in range(90000,90200):
        episode_indices=[lookup[(episode_seed,k)] for k in range(sum(seed==episode_seed))]
        ft,history=ukf_episode(model,obs[episode_indices],act[episode_indices],Q,R)
        diag.extend((episode_seed,*row) for row in history)
        for k in range(len(episode_indices)):
            i=row_lookup.get((episode_seed,k))
            if i is not None:output[i]=ft[k]
        if episode_seed%25==0:print(f"G9 UKF filtered seed {episode_seed}",flush=True)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("UKF output invalid")
    np.savez_compressed(OUT/"ukf_current_features.npz",features=output)
    json_new(OUT/"filter_definition.json",{"filter":"unscented Kalman filter on 10 normalized 2D tracking positions + 10 per-step velocities",
        "formulation":"2n+1 sigma points, alpha=1 beta=2 kappa=0; nonlinear train-fitted transition, direct position measurement correction",
        "transition_checkpoint":"models/transition.pt","process_covariance_diagonal":Q.tolist(),
        "measurement_covariance_diagonal":[R]*10,"initial_position_variance":R,
        "initial_velocity_variance":0.04,"state_reset":"per episode from first measured tracking positions and zero velocity",
        "feature_output":"current 14D observation + corrected 10D velocity + log posterior velocity covariance diagonal",
        "reason_UKF":"guidewire tracking evolution depends nonlinearly on action and shape; sigma-point propagation handles nonlinear learned process without linearizing it; measurement is current 2D tracking only",
        "no_future_measurement_correction":True,"no_force_sensor":True,
        "training_source":"G3 train episodes only for nonlinear transition and process covariance"})
    arr=np.asarray(diag,dtype=np.float64)
    json_new(OUT/"filter_diagnostics.json",{"steps":len(arr),"innovation_norm_mean":float(np.mean(arr[:,1])),
        "innovation_norm_q90":float(np.quantile(arr[:,1],.9)),
        "minimum_pre_projection_covariance_eigenvalue":float(np.min(arr[:,2])),
        "maximum_posterior_diagonal_variance":float(np.max(arr[:,3])),
        "nonfinite_count":int(np.sum(~np.isfinite(arr)))})
    print(f"G9 UKF current features {output.shape}; finite={np.all(np.isfinite(output))}",flush=True)


def feature_views(d):
    obs=d["history_observation"]
    return {"E0":obs[:,2],
        "E1":np.concatenate((obs.reshape(len(obs),42),d["history_action"].reshape(len(obs),4)),axis=1),
        "E2":np.load(OUT/"ukf_current_features.npz")["features"]}


def train_mappers():
    cuda_gate();d=data();views=feature_views(d)
    tr,va=split_mask(d,"train"),split_mask(d,"validation")
    target=d["privileged_physical"].astype(np.float64)
    ymean=np.mean(target[tr],axis=0);ystd=np.std(target[tr],axis=0);ystd[ystd<1e-3]=1e-3
    for j,(name,x) in enumerate(views.items()):
        folder=OUT/"models"/name
        folder.mkdir(parents=True,exist_ok=False)
        x=np.asarray(x,dtype=np.float64)
        mean=np.mean(x[tr],axis=0);scale=np.std(x[tr],axis=0);scale[scale<1e-4]=1.
        json_new(folder/"normalization.json",{"input_mean":mean.tolist(),"input_std":scale.tolist(),
            "target_mean":ymean.tolist(),"target_std":ystd.tolist(),"train_only":True})
        tx=torch.as_tensor((x[tr]-mean)/scale,dtype=torch.float32,device=DEVICE)
        ty=torch.as_tensor((target[tr]-ymean)/ystd,dtype=torch.float32,device=DEVICE)
        vx=torch.as_tensor((x[va]-mean)/scale,dtype=torch.float32,device=DEVICE)
        vy=torch.as_tensor((target[va]-ymean)/ystd,dtype=torch.float32,device=DEVICE)
        torch.manual_seed(SEED+j);net=Mapper(x.shape[1]).to(DEVICE)
        optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
        generator=torch.Generator(device="cpu").manual_seed(SEED+j)
        best=math.inf;patience=0;history=[]
        for epoch in range(75):
            net.train();order=torch.randperm(len(tx),generator=generator).to(DEVICE)
            for chunk in order.split(512):
                loss=(net(tx[chunk])-ty[chunk]).square().mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
            net.eval()
            with torch.inference_mode():val=float((net(vx)-vy).square().mean().cpu())
            history.append({"epoch":epoch,"validation_normalized_MSE":val})
            if val<best-1e-6:
                best=val;patience=0;torch.save(net.cpu().state_dict(),folder/"model.pt");net.to(DEVICE)
            else:
                patience+=1
                if patience>=8:break
        json_new(folder/"training.json",{"best_validation_normalized_MSE":best,"history":history,
            "training_seed":SEED+j,"selection":"validation only"})
        print(f"G9 {name} P mapper validation normalized MSE {best:.6g}",flush=True)


def predict_mappers():
    cuda_gate();d=data();views=feature_views(d)
    preds={}
    for name,x in views.items():
        norm=json.loads((OUT/"models"/name/"normalization.json").read_text())
        x=(np.asarray(x,dtype=np.float64)-norm["input_mean"])/norm["input_std"]
        net=Mapper(x.shape[1]).to(DEVICE)
        net.load_state_dict(torch.load(OUT/"models"/name/"model.pt",map_location=DEVICE,weights_only=True));net.eval()
        with torch.inference_mode():
            p=np.concatenate([net(torch.as_tensor(block,dtype=torch.float32,device=DEVICE)).cpu().numpy()
                for block in np.array_split(x,math.ceil(len(x)/2048))],axis=0)
        preds[name]=(p*np.asarray(norm["target_std"])+np.asarray(norm["target_mean"])).astype(np.float32)
    np.savez_compressed(OUT/"estimated_current_physical.npz",**preds)
    rows=[];truth=d["privileged_physical"]
    groups={"tip_position_3d_mm":slice(0,3),"distal_shape_offsets_mm":slice(3,15),
        "distal_velocity_mm_per_s":slice(15,30),"contact_force_vector_N":slice(30,33),
        "contact_magnitude_N":slice(33,34),"contact_count":slice(34,35),
        "insertion_length_mm":slice(35,36),"rotation_sin_cos":slice(36,38)}
    for name,p in preds.items():
        for group in ("train","validation","development_test"):
            mask=split_mask(d,group)
            for component,sl in groups.items():
                error=p[mask,sl]-truth[mask,sl]
                rows.append({"estimator":name,"group":group,"component":component,"rows":int(np.sum(mask)),
                    "RMSE":float(np.sqrt(np.mean(error**2))),"MAE":float(np.mean(np.abs(error))),
                    "bias":float(np.mean(error))})
    csv_new(OUT/"state_estimation_metrics.csv",rows)
    print("G9 E0/E1/E2 privileged-current-state estimates saved",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("transition","filter","mappers","predict"),required=True)
    stage=parser.parse_args().stage
    {"transition":train_transition,"filter":filter_all,"mappers":train_mappers,
     "predict":predict_mappers}[stage]()


if __name__=="__main__":main()
