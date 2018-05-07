from qd_common import load_net
from caffe import layers as L, params as P, to_proto

class TemplateNet(object):
    def add_body(self, n, **kwargs):
        '''
        we add a dummy layer, and this layer will be removed and populated by
        the real net
        '''
        n['last_ip'] = L.InnerProduct(n.data, num_output=1000)
