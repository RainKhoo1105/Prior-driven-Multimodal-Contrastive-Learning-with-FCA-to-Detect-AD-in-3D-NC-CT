from train_newsets_fftloss_attention import SimCLR
import yaml

def main():
    config = yaml.load(open("config_newsets_fftloss_attention.yaml", "r"), Loader=yaml.FullLoader)
    
    self_wrf=config["fca_loss_weight"]*10

    simclr = SimCLR(config, gpu_id=[0],modelname=f"fca_{self_wrf[0]}{self_wrf[1]}")#weight=[text_weight,cta_weight]
    simclr.train()


if __name__ == "__main__":
    main()
