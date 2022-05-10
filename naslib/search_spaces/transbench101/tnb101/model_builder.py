from .models.decoder import GenerativeDecoder
from .models.discriminator import Discriminator
from .models.feedforward import FeedForwardNet
from .models.gan import GAN
from .models.segmentation import Segmentation
from .models.siamese import SiameseNet
from .models.decoder import FFDecoder, SegmentationDecoder, SiameseDecoder
from .models.encoder import FFEncoder


def create_model(encoder_str, task_name):
    cfg = {}

    # basics
    cfg['encoder_str'] = encoder_str
    cfg['task_name'] = task_name

    # model
    cfg['encoder'] = FFEncoder(encoder_str, task_name=cfg['task_name']).network
    cfg['decoder_input_dim'] = (2048, 16, 16) if cfg['encoder_str'] == 'resnet50' else cfg['encoder'].output_dim

    if task_name == 'segmentsemantic':
        model = _create_model_segmentsemantic(cfg)
    elif task_name == 'class_object':
        model = _create_model_class_object(cfg)
    elif task_name == 'class_scene':
        model = _create_model_class_scene(cfg)
    elif task_name == 'jigsaw':
        model = _create_model_jigsaw(cfg)
    elif task_name == 'room_layout':
        model = _create_model_room_layout(cfg)
    elif task_name == 'autoencoder':
        model = _create_model_autoencoder(cfg)
    elif task_name == 'normal':
        model = _create_model_normal(cfg)
    else:
        raise NotImplementedError(f'Model not implemented for task {task_name}')

    return model

def _create_feed_forward_net(cfg):
    cfg['decoder'] = FFDecoder(cfg['decoder_input_dim'], cfg['target_dim'])
    cfg['model'] = FeedForwardNet(cfg['encoder'], cfg['decoder'])
    return cfg['model']

def _create_model_segmentsemantic(cfg):
    cfg['target_dim'] = (256, 256)
    cfg['target_num_channel'] = 17
    cfg['decoder'] = SegmentationDecoder(cfg['decoder_input_dim'], cfg['target_dim'],
                                         target_num_channel=cfg['target_num_channel'])
    cfg['model'] = Segmentation(cfg['encoder'], cfg['decoder'])

    return cfg['model']

def _create_model_class_object(cfg):
    cfg['target_dim'] = 100 # ORIG CODE 100
    return _create_feed_forward_net(cfg)

def _create_model_class_scene(cfg):
    cfg['target_dim'] = 63 # ORIG CODE 63
    return _create_feed_forward_net(cfg)

def _create_model_jigsaw(cfg):
    cfg['target_dim'] = 1000
    cfg['decoder'] = SiameseDecoder(cfg['decoder_input_dim'], cfg['target_dim'], num_pieces=9)
    cfg['model'] = SiameseNet(cfg['encoder'], cfg['decoder'])

    return cfg['model']

def _create_model_room_layout(cfg):
    cfg['target_dim'] = 9
    return _create_feed_forward_net(cfg)

def _create_model_autoencoder(cfg):
    cfg['target_dim'] = (256, 256) # ORIG CODE (3, 256, 256)
    cfg['decoder'] = GenerativeDecoder(cfg['decoder_input_dim'], cfg['target_dim'])
    cfg['discriminator'] = Discriminator()
    cfg['model'] = GAN(cfg['encoder'], cfg['decoder'], cfg['discriminator'])

    return cfg['model']

def _create_model_normal(cfg):
    return _create_model_autoencoder(cfg)
