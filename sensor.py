from network import PNN
from utils import net_scope


class Sensor:
    def __init__(self, sensor):

        self.sensor = sensor
        self.ratio = 4

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'WV2') or (sensor == 'WV3'):
            self.kernels = [9, 5, 5]
        elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            self.kernels = [5, 5, 5]

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            self.nbands = 4
        elif (sensor == 'WV2') or (sensor == 'WV3'):
            self.nbands = 8
        self.net_scope = net_scope(self.kernels)
        self.nbits = 11

        self.net = PNN(self.nbands+1, self.kernels, self.net_scope)





if __name__ == '__main__':
    import torch
    PATH = '/home/matteo/Scrivania/PNN/pnn_pytorch_release_v3/Testing/pretrained_models/WV3_PNN_noIDX_model.pth.tar'
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    s = Sensor('WV3')
    chk = torch.load(PATH, map_location=torch.device('cpu'))
    s.net.load_state_dict(chk['model_state'])
    s.net = s.net.float()

    save_path = '/home/matteo/PycharmProjects/PNN/weights/WV3_PNN_noIDX_model.tar'
    torch.save(s.net.state_dict(), save_path)

    """
    s.net.to(device)

    inp = torch.ones((1,s.nbands+1, 20,20))
    inp = inp.to(device)
    s.net.zero_grad()
    out = s.net(inp)

    output = out.cpu()
    output = output.detach().numpy()
    input = inp.cpu().numpy()
    """