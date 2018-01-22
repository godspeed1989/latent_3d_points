import numpy as np
import tensorflow as tf
from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.normalization import batch_normalization

blue = lambda x:'\033[94m' + x + '\033[0m'

'''
    工具函数
'''
def expand_scope_by_name(scope, name):
    """ expand tf scope by given name.
    """
    if isinstance(scope, str):
        scope += '/' + name
        return scope

    if scope is not None:
        return scope.name + '/' + name
    else:
        return scope

def replicate_parameter_for_all_layers(parameter, n_layers):
    if parameter is not None and len(parameter) != n_layers:
        if len(parameter) != 1:
            raise ValueError()
        parameter = np.array(parameter)
        parameter = parameter.repeat(n_layers).tolist()
    return parameter

'''
    AutoEncoder 编码器结构
'''
def encoder_with_convs_and_symmetry_new(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, scope=None,
                                        reuse=False, padding='same', closing=None, conv_op=conv_1d):
    '''
    An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    print(blue('Building Encoder'), blue(str(n_filters)))

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)
        print(name, 'conv params =', layer.W.get_shape().as_list(), layer.b.get_shape().as_list())

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            print('bnorm params =', layer.beta.get_shape().as_list(), layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        print(layer)
        print('output size:', layer.get_shape().as_list(), '\n')

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        print(layer)

    if closing is not None:
        layer = closing(layer)
        print(layer)

    return layer

'''
    AutoDecoder 解码器结构
'''
def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    print(blue('Building Decoder'), blue(str(layer_sizes)))

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in range(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
        print(name, 'FC params = ', layer.W.get_shape().as_list(), layer.b.get_shape().as_list())

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            print('bnorm params = ', layer.beta.get_shape().as_list(), layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        print(layer)
        print('output size:', layer.get_shape().as_list(), '\n')

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    print(name, 'FC (last) params = ', layer.W.get_shape().as_list(), layer.b.get_shape().as_list())

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        print('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

    print(layer)
    print('output size:', layer.get_shape().as_list(), '\n')

    return layer

'''
    AutoEncoder总体结构
'''
def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, bneck_post_mlp=False):
    encoder = encoder_with_convs_and_symmetry_new
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

