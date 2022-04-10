import torch
import pandas as pd
import models_and_tools as t
import numpy as np

res = t.Resnet().cuda().eval()
cnn = t.CNN().cuda().eval()
rnn = t.RNN().cuda().eval()

net_dict = {cnn: 'cnn_model_cnn.pth.tar',
            rnn: 'cnn_model_rnn.pth.tar',
            res: 'cnn_model_res.pth.tar'}

file_dict = {
    0: 'down',
    1: 'sit',
    2: 'stand',
    3: 'up',
    4: 'walk'
}
def get_m_and_v(number, axis):
    temp = t.column_to_row('txt/'+file_dict[number]+'.txt')
    mean = np.mean(temp[axis])
    var = np.var(temp[axis])
    return mean, var

def detect(number, net_form):
    net = net_form
    f = pd.read_excel('detect/'+file_dict[number]+'.xls', skiprows=1)
    a = f.to_numpy().transpose(1, 0)
    x_mean, x_var = get_m_and_v(number, 0)
    y_mean, y_var = get_m_and_v(number, 1)
    z_mean, z_var = get_m_and_v(number, 2)
    x = (a[0] - x_mean)/x_var
    y = (a[1] - y_mean)/y_var
    z = (a[2] - z_mean)/z_var
    b = np.array([x, y, z], dtype=np.float32)
    a_x = t.window_slide(b[0], 128, 64)
    a_y = t.window_slide(b[1], 128, 64)
    a_z = t.window_slide(b[2], 128, 64)
    out = t.unit_axis(a_x, a_y, a_z)
    out = torch.from_numpy(out).cuda()

    load = torch.load(net_dict[net_form])
    net.load_state_dict(load['state_dict'])

    out = net(out)
    _, pre = torch.max(out, 1)
    print(pre)

for i in range(5):
    detect(i, res)

