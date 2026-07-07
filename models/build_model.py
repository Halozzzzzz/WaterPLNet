from models.networks import WaterPLNet


def build_model(opt, flag='train'):
    print(f"==> Building model: {opt.model_name}")
    
    if opt.model_name == 'WaterPLNet':
        return WaterPLNet(opt, flag)
    
    else:
        raise NotImplementedError(f"Model {opt.model_name} not found!")
