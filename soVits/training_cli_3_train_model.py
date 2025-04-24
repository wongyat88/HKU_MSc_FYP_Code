import json, yaml
import sys
import os
import torch
import psutil
from subprocess import Popen
from tools.my_utils import check_for_existance, check_details
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share

version = "v3"

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False
ok_gpu_keywords={"10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060","H","600","506","507","508","509"}
set_gpu_numbers=set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

def set_default():
    global default_batch_size,default_max_batch_size,gpu_info,default_sovits_epoch,default_sovits_save_every_epoch,max_sovits_epoch,max_sovits_save_every_epoch,default_batch_size_s1,if_force_ckpt
    if_force_ckpt = False
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        minmem = min(mem)
        if version == "v3" and minmem < 14:
            # API读取不到共享显存,直接填充确认
            try:
                torch.zeros((1024,1024,1024,14),dtype=torch.int8,device="cuda")
                torch.cuda.empty_cache()
                minmem = 14
            except RuntimeError as _:
                # 强制梯度检查只需要12G显存
                if minmem >= 12 :
                    if_force_ckpt = True
                    minmem = 14
                else:
                    try:
                        torch.zeros((1024,1024,1024,12),dtype=torch.int8,device="cuda")
                        torch.cuda.empty_cache()
                        if_force_ckpt = True
                        minmem = 14
                    except RuntimeError as _:
                        print("显存不足以开启V3训练")
        default_batch_size = minmem // 2 if version!="v3"else minmem//8
        default_batch_size_s1=minmem // 2
    else:
        gpu_info = ("%s\t%s" % ("0", "CPU"))
        gpu_infos.append("%s\t%s" % ("0", "CPU"))
        set_gpu_numbers.add(0)
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 4)
    if version!="v3":
        default_sovits_epoch=8
        default_sovits_save_every_epoch=4
        max_sovits_epoch=25#40
        max_sovits_save_every_epoch=25#10
    else:
        default_sovits_epoch=2
        default_sovits_save_every_epoch=1
        max_sovits_epoch=3#40
        max_sovits_save_every_epoch=3#10

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3

set_default()

gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers=str(sorted(list(set_gpu_numbers))[0])
def fix_gpu_number(input):#将越界的number强制改到界内
    try:
        if(int(input)not in set_gpu_numbers):return default_gpu_numbers
    except:return input
    return input
def fix_gpu_numbers(inputs):
    output=[]
    try:
        for input in inputs.split(","):output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
SoVITS_weight_root = ["SoVITS_weights","SoVITS_weights_v2","SoVITS_weights_v3"]
p_train_SoVITS = None


def start_sovits_training(batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D,if_grad_ckpt,lora_rank,s2_dir):
    global p_train_SoVITS
    if(p_train_SoVITS==None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir=s2_dir #  "%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s2_%s"%(s2_dir,version),exist_ok=True)
        if check_for_existance([s2_dir],is_train=True):
            check_details([s2_dir],is_train=True)
        if(is_half==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,batch_size//2)
        print(batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D,if_grad_ckpt,lora_rank)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["train"]["grad_ckpt"]=if_grad_ckpt
        data["train"]["lora_rank"]=lora_rank
        data["model"]["version"]=version
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root[int(version[-1])-1]
        data["name"]=exp_name
        data["version"]=version
        tmp_config_path="%s/tmp_s2.json"%tmp
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))
        if version in ["v1","v2"]:
            cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%(python_exec,tmp_config_path)
        else:
            cmd = '"%s" GPT_SoVITS/s2_train_v3_lora.py --config "%s"'%(python_exec,tmp_config_path)
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        print("SOVITS training finished")


p_train_GPT=None
GPT_weight_root=["GPT_weights","GPT_weights_v2","GPT_weights_v3"]


def start_gpt_training(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1,s1_dir):
    global p_train_GPT
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml"if version=="v1"else "GPT_SoVITS/configs/s1longer-v2.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir= s1_dir #  "%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        if check_for_existance([s1_dir],is_train=True):
            check_details([s1_dir],is_train=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        print(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_dpo"]=if_dpo
        data["train"]["half_weights_save_dir"]=GPT_weight_root[int(version[-1])-1]
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1_%s"%(s1_dir,version)
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"]= fix_gpu_numbers(gpu_numbers.replace("-",","))
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%tmp
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        print("GPT training finished")

if __name__ == '__main__':
    exp_dir = "Temp_20250422_v3"
    exp_name="tmp"

    # SOVITS Training Parameters
    s_batch_size = 11
    s_epoch = 8
    s_exp_name = exp_name
    s_text_low_lr_rate = 0.4
    s_if_save_latest = True
    s_if_save_every_weights = True
    s_save_every_n_epoch = 4
    s_gpunumbers = "0"
    s_pretrained_s2G = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    s_pretrained_s2D = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    s_if_grad_ckpt = False
    s_lora_rank = 32
    s2_dir = f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/{s_exp_name}"

    start_sovits_training(s_batch_size,
                          s_epoch,
                          s_exp_name,
                          s_text_low_lr_rate,
                          s_if_save_latest,
                          s_if_save_every_weights,
                          s_save_every_n_epoch,
                          s_gpunumbers,
                          s_pretrained_s2G,
                          s_pretrained_s2D,
                          s_if_grad_ckpt,
                          s_lora_rank,
                          s2_dir)

    # start_sovits_training(11,8,"20250402_Test04",0.4,True,True,4,"0","L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth","L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",False,32,"L:/MSCCS Project/GPT-SoVITS/Experimental/Steps/4_Preprocessing/20250402_Test04")


    # GPT Training Parameters
    g_batch_size = 11
    g_epoch = 15
    g_exp_name = exp_name
    g_if_dpo = False
    g_if_save_latest = True
    g_if_save_every_weights = True
    g_save_every_n_epoch = 5
    g_gpu_numbers = "0"
    g_pretrained_s1 = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    s1_dir = f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/{s_exp_name}"

    start_gpt_training(g_batch_size,
                       g_epoch,
                       g_exp_name,
                       g_if_dpo,
                       g_if_save_latest,
                       g_if_save_every_weights,
                       g_save_every_n_epoch,
                       g_gpu_numbers,
                       g_pretrained_s1,
                       s1_dir)


    # start_gpt_training(11,15,"20250402_Test04",False,True,True,5,"0","L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt","L:/MSCCS Project/GPT-SoVITS/Experimental/Steps/4_Preprocessing/20250402_Test04")
