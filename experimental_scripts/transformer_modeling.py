from torch.utils.tensorboard import SummaryWriter
import pandas as pd

logger = SummaryWriter("./temps/logs")

train = pd.read_csv('./temps/optiver/train.csv')
revealed_targets = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
test = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv')
sample_submission = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/example_test_files/sample_submission.csv')