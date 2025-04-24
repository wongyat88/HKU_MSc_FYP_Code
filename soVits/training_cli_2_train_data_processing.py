import os
import torch
import psutil
from tools import my_utils
from tools.my_utils import load_audio, check_for_existance, check_details
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
from subprocess import Popen

version = "v2"

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
        # if version == "v3" and minmem < 14:
        #     # API读取不到共享显存,直接填充确认
        #     try:
        #         torch.zeros((1024,1024,1024,14),dtype=torch.int8,device="cuda")
        #         torch.cuda.empty_cache()
        #         minmem = 14
        #     except RuntimeError as _:
        #         # 强制梯度检查只需要12G显存
        #         if minmem >= 12 :
        #             if_force_ckpt = True
        #             minmem = 14
        #         else:
        #             try:
        #                 torch.zeros((1024,1024,1024,12),dtype=torch.int8,device="cuda")
        #                 torch.cuda.empty_cache()
        #                 if_force_ckpt = True
        #                 minmem = 14
        #             except RuntimeError as _:
        #                 print("显存不足以开启V3训练")
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


ps1abc=[]

def preprocess_one_step(inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G_path,opt_dir):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text,inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text,inp_wav_dir], is_dataset_processing=True)
    if (ps1abc == []):
        opt_dir= opt_dir #  "%s/%s"%(exp_root,exp_name)
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print("1A-Doing")
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                # assert len("".join(opt)) > 0, process_info(process_name_1a, "failed")
            print("1A-Done")
            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":ssl_pretrained_dir,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            print("1B-Doing")
            for p in ps1abc:p.wait()
            print("1B-Done")
            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G_path,
                    "s2config_path":"GPT_SoVITS/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print("1C-Doing")
                for p in ps1abc:p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print("1C-Done")
            ps1abc = []
            print("Finished")
        except:
            print("Failed")
    else:
        print("Occupied")

if __name__ == '__main__':
    exp_dir = "Temp_20250422_v3"
    exp_name="tmp"
    preprocess_one_step(inp_text=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/3_asr_out/2_denoise_out.list",
                        inp_wav_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/2_denoise_out",
                        exp_name=exp_name,
                        gpu_numbers1a="0-0",
                        gpu_numbers1Ba="0-0",
                        gpu_numbers1c="0-0",
                        bert_pretrained_dir="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                        ssl_pretrained_dir="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base",
                        pretrained_s2G_path="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                        opt_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/{exp_name}")