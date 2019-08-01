import torch
import string
import torch.utils.data
import pandas as pd
import torch.backends.cudnn as cudnn

from models.deepTextRecognitionBenchmark.utils import CTCLabelConverter, AttnLabelConverter
from models.deepTextRecognitionBenchmark.dataset import RawDataset, AlignCollate
from models.deepTextRecognitionBenchmark.model import Model

from typing import List


class TextRecognition(object):

    def __init__(self, saved_model: str, rgb, PAD, Transformation: str, FeatureExtraction: str,
                 SequenceModeling: str,  Prediction: str, num_fiducial: int = 20,
                 input_channel: int = 1, output_channel: int = 512, hidden_size: int = 256,
                 workers: int = 4, batch_size: int = 192,
                 batch_max_length: int = 25, imgH: int = 32, imgW: int = 100, character: str = '0123456789abcdefghijklmnopqrstuvwxyz',
                 sensitive: bool = True, gpu: bool = False):
        """
        :param saved_model       :  path to saved_model to evaluation
        :param rgb               : use rgb input
        :param PAD               : whether to keep ratio then pad for image resize
        :param Transformation    : Transformation stage. None|TPS
        :param FeatureExtraction : FeatureExtraction stage. VGG|RCNN|ResNet
        :param SequenceModeling  : SequenceModeling stage. None|BiLSTM
        :param Prediction        : Prediction stage. CTC|Attn
        :param num_fiducial      : number of fiducial points of TPS-STN
        :param input_channel     : the number of input channel of Feature extractor
        :param output_channel    : the number of output channel of Feature extractor
        :param hidden_size       : the size of the LSTM hidden state
        :param workers           : number of data loading workers
        :param batch_size        : input batch size
        :param batch_max_length  : maximum-label-length
        :param imgH              : the height of the input image
        :param imgW              : the width of the input image
        :param character         : character label
        :param sensitive         : for sensitive character mode
        """
        self.saved_model = saved_model
        self.rgb = rgb
        self.PAD = PAD
        self.Transformation = Transformation
        self.FeatureExtraction = FeatureExtraction
        self.SequenceModeling = SequenceModeling
        self.Prediction = Prediction
        self.num_fiducial = num_fiducial
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_size = hidden_size
        self.workers = workers
        self.batch_size = batch_size
        self.batch_max_length = batch_max_length
        self.imgH = imgH
        self.imgW = imgW
        self.character = character
        self.sensitive = sensitive
        self.gpu = gpu

        """ vocab / character number configuration """
        if self.sensitive:
            self.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        if self.gpu:
            cudnn.benchmark = True
            cudnn.deterministic = True
            self.num_gpu = torch.cuda.device_count()

        if 'CTC' in self.Prediction:
            self.converter = CTCLabelConverter(self.character)
        else:
            self.converter = AttnLabelConverter(self.character)
        self.num_class = len(self.converter.character)

        if self.rgb:
            self.input_channel = 3
        self.model = Model(opt=self)

        self.model = torch.nn.DataParallel(self.model)
        if torch.cuda.is_available():
            model = self.model.cuda()

        # load model
        print(f'loading pretrained model from {self.saved_model}...')
        self.model.load_state_dict(torch.load(self.saved_model, map_location='cpu'))

    def prepare_data(self, images: List[str]):
        """
        :param images: List of paths to the images
        """
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD)
        demo_data = RawDataset(image_path_list=images, opt=self)  # use RawDataset
        return torch.utils.data.DataLoader(
            demo_data, batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(self.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

    def recognize(self, images: List[str]):
        """
        :param images: List of paths to the images
        """
        self.model.eval()

        for image_tensors, image_path_list in self.prepare_data(images=images):
            batch_size = image_tensors.size(0)
            with torch.no_grad():
                if self.gpu:
                    image = image_tensors.cuda()
                else:
                    image = image_tensors
                # For max length prediction
                if self.gpu:
                    length_for_pred = torch.cuda.IntTensor([self.batch_max_length] * batch_size)
                    text_for_pred = torch.cuda.LongTensor(batch_size, self.batch_max_length + 1).fill_(0)
                else:
                    length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size)
                    text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0)

            if 'CTC' in self.Prediction:
                preds = self.model(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            result = pd.DataFrame(list(zip(image_path_list, preds_str)), columns=['image_path', 'pred'])
            if 'Attn' in self.Prediction:
                result['pred'] = result['pred'].apply(lambda x: x[:x.find('[s]')])  # prune after "end of sentence" token ([s])

            return result
