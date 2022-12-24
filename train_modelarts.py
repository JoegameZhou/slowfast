#  
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Train workflow on ModelArts."""
import os
import moxing as mox
import shutil
from mindspore import context, nn, dtype, load_checkpoint, set_seed
from mindspore import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from src.config.defaults import assert_and_infer_cfg
from src.utils.parser import load_config, parse_args
from src.utils import logging
from src.datasets.build import build_dataset
from src.models.video_model_builder import SlowFast
from src.models import optimizer as optim

set_seed(42)

class NetWithLoss(nn.Cell):
    """Construct Loss Net."""
    def __init__(self, net):
        super().__init__()
        self.loss = nn.BCELoss(reduction='none').to_float(dtype.float32)
        self.net = net.to_float(dtype.float32)

    def construct(self, slowpath, fastpath, boxes, labels, mask):
        preds = self.net(slowpath, fastpath, boxes)
        # (n * max_num, class) -> (n, max_num, class)
        preds = preds.reshape(mask.shape + (-1,))
        # (n, max_num) -> (n, max_num, 1)
        mask = mask.reshape(mask.shape + (1,))
        loss = self.loss(preds, labels) * mask
        loss = loss.astype(dtype.float32).sum() / mask.astype(dtype.float32).sum() / preds.shape[2]
        return loss

def train():
    """Train entrance."""
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    ######################## 将数据集从obs拷贝到训练镜像中 （固定写法）########################   
    # 在训练环境中定义data_url和train_url，并把数据从obs拷贝到相应的固定路径
    workroot = '/home/work/user-job-dir'
    data_dir = workroot + '/data'  #数据集存放路径
    train_dir = workroot + '/model' #模型存放路径
    #初始化数据存放目录
    # if os.path.exists(data_dir):
    #     shutil.rmtree(data_dir)
    # os.mkdir(data_dir)
    #初始化模型存放目录
    obs_train_url = args.train_url
    train_dir = workroot + '/model/'
    if os.path.exists(train_dir):
        os.mkdir(train_dir)
    obs_data_url = args.data_url
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url,
                                                      data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(
            obs_data_url, data_dir) + str(e))
    ######################## 将数据集从obs拷贝到训练镜像中 ########################
    # setup logger
    logger = logging.get_logger(__name__)
    logging.setup_logging()
    logger.info(cfg)
    # setup context
    init()
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('DEVICE_NUM', '1'))
    # On modelarts.
    device_num, rank_id = _get_rank_info()
    context.set_context(device_id=device_id, mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(save_graphs=True, save_graphs_path='irs')
    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    logger.info("rank_id = %d, device_num = %d", rank_id, device_num)
    # build dataset
    dataset = build_dataset(cfg, "train", num_shards=device_num, shard_id=rank_id)
    steps_per_epoch = dataset.get_dataset_size()
    # build net with loss
    network = SlowFast(cfg).set_train(True)
    net_with_loss = NetWithLoss(network).to_float(dtype.float32)
    # load ckpt
    print(f'loading {cfg.TRAIN.CHECKPOINT_FILE_PATH}')
    param_dict = load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, net_with_loss)

    # build optimizer
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=1024, scale_window=1000)
    optimizer = optim.construct_optimizer(net_with_loss, steps_per_epoch, cfg)

    # setup callbacks
    callbacks = [TimeMonitor(), LossMonitor()]
    if (device_num == 1) or (device_num > 1 and device_id in [0, 7]):
        ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=cfg.SOLVER.MAX_EPOCH)
        ckpt_cb = ModelCheckpoint(prefix="slowfast", directory=train_dir, config=ckpt_cfg)
        callbacks.append(ckpt_cb)

    # build model
    model = Model(network=net_with_loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager)
    # start training
    logger.info("============== Starting Training ==============")
    logger.info(f"total_epoch={cfg.SOLVER.MAX_EPOCH}, steps_per_epoch={steps_per_epoch}")
    model.train(cfg.SOLVER.MAX_EPOCH, dataset, callbacks=callbacks, dataset_sink_mode=True)
    ######################## 将输出的模型拷贝到obs（固定写法） ########################   
    # 把训练后的模型数据从本地的运行环境拷贝回obs，在启智平台相对应的训练任务中会提供下载
    train_dir = workroot + '/model/'
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,
                                                    obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,
                                                    obs_train_url) + str(e))
    ######################## 将输出的模型拷贝到obs ######################## 

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id

if __name__ == "__main__":
    train()