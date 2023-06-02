# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import torchvision.models

import trojai.modelgen.architecture_factory

ALL_ARCHITECTURE_KEYS = ['resnet18','resnet34','resnet50','resnet101','resnet152','googlenet','inceptionv3','densenet121','densenet161','densenet169','densenet201','squeezenetv1_0','squeezenetv1_1','wideresnet50','wideresnet101','mobilenetv2','mnasnet0_5','mnasnet0_75','mnasnet1_0','mnasnet1_3','shufflenet1_0','shufflenet1_5','shufflenet2_0','vgg11bn','vgg13bn','vgg16bn','vgg19bn']

architecture_keys = ['resnet18','resnet34','resnet50','resnet101','googlenet','inceptionv3','densenet121','squeezenetv1_0','squeezenetv1_1','wideresnet50','mobilenetv2','shufflenet1_0','shufflenet1_5','shufflenet2_0','vgg11bn','vgg13bn']


class Resnet18ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.resnet18(pretrained=False, progress=True, num_classes=number_classes)


class Resnet34ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.resnet34(pretrained=False, progress=True, num_classes=number_classes)


class Resnet50ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.resnet50(pretrained=False, progress=True, num_classes=number_classes)


class Resnet101ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.resnet101(pretrained=False, progress=True, num_classes=number_classes)


class Resnet152ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.resnet152(pretrained=False, progress=True, num_classes=number_classes)


class GoogLeNetArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.googlenet(pretrained=False, progress=True, num_classes=number_classes, aux_logits=False)


class InceptionV3ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=number_classes, aux_logits=False)


class DenseNet121ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.densenet121(pretrained=False, progress=True, num_classes=number_classes)


class DenseNet161ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.densenet161(pretrained=False, progress=True, num_classes=number_classes)


class DenseNet169ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.densenet169(pretrained=False, progress=True, num_classes=number_classes)


class DenseNet201ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.densenet201(pretrained=False, progress=True, num_classes=number_classes)


class SqueezeNetV1_0ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.squeezenet1_0(pretrained=False, progress=True, num_classes=number_classes)


class SqueezeNetV1_1ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.squeezenet1_1(pretrained=False, progress=True, num_classes=number_classes)


class WideResNet50ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.wide_resnet50_2(pretrained=False, progress=True, num_classes=number_classes)


class WideResNet101ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.wide_resnet101_2(pretrained=False, progress=True, num_classes=number_classes)


class MobileNetV2ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.mobilenet_v2(pretrained=False, progress=True, num_classes=number_classes)


class MNASNet0_5ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.mnasnet0_5(pretrained=False, progress=True, num_classes=number_classes)


class MNASNet0_75ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.mnasnet0_75(pretrained=False, progress=True, num_classes=number_classes)


class MNASNet1_0ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.mnasnet1_0(pretrained=False, progress=True, num_classes=number_classes)


class MNASNet1_3ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.mnasnet1_3(pretrained=False, progress=True, num_classes=number_classes)


class ShuffleNet0_5ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.shufflenet_v2_x0_5(pretrained=False, progress=True, num_classes=number_classes)


class ShuffleNet1_0ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.shufflenet_v2_x1_0(pretrained=False, progress=True, num_classes=number_classes)


class ShuffleNet1_5ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.shufflenet_v2_x1_5(pretrained=False, progress=True, num_classes=number_classes)


class ShuffleNet2_0ArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.shufflenet_v2_x2_0(pretrained=False, progress=True, num_classes=number_classes)


class VGG11NetArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.vgg11_bn(pretrained=False, progress=True, num_classes=number_classes)


class VGG13NetArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.vgg13_bn(pretrained=False, progress=True, num_classes=number_classes)


class VGG16NetArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.vgg16_bn(pretrained=False, progress=True, num_classes=number_classes)


class VGG19NetArchitectureFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, number_classes: int):
        return torchvision.models.vgg19_bn(pretrained=False, progress=True, num_classes=number_classes)


def get_factory(model_name: str):
    model = None

    if model_name == 'resnet18':
        model = Resnet18ArchitectureFactory()
    elif model_name == 'resnet34':
        model = Resnet34ArchitectureFactory()
    elif model_name == 'resnet50':
        model = Resnet50ArchitectureFactory()
    elif model_name == 'resnet101':
        model = Resnet101ArchitectureFactory()
    elif model_name == 'resnet152':
        model = Resnet152ArchitectureFactory()
    elif model_name == 'googlenet':
        model = GoogLeNetArchitectureFactory()
    elif model_name == 'inceptionv3':
        model = InceptionV3ArchitectureFactory()
    elif model_name == 'densenet121':
        model = DenseNet121ArchitectureFactory()
    elif model_name == 'densenet161':
        model = DenseNet161ArchitectureFactory()
    elif model_name == 'densenet169':
        model = DenseNet169ArchitectureFactory()
    elif model_name == 'densenet201':
        model = DenseNet201ArchitectureFactory()
    elif model_name == 'squeezenetv1_0':
        model = SqueezeNetV1_0ArchitectureFactory()
    elif model_name == 'squeezenetv1_1':
        model = SqueezeNetV1_1ArchitectureFactory()
    elif model_name == 'wideresnet50':
        model = WideResNet50ArchitectureFactory()
    elif model_name == 'wideresnet101':
        model = WideResNet101ArchitectureFactory()
    elif model_name == 'mobilenetv2':
        model = MobileNetV2ArchitectureFactory()
    elif model_name == 'mnasnet0_5':
        model = MNASNet0_5ArchitectureFactory()
    elif model_name == 'mnasnet0_75':
        model = MNASNet0_75ArchitectureFactory()
    elif model_name == 'mnasnet1_0':
        model = MNASNet1_0ArchitectureFactory()
    elif model_name == 'mnasnet1_3':
        model = MNASNet1_3ArchitectureFactory()
    elif model_name == 'shufflenet1_0':
        model = ShuffleNet1_0ArchitectureFactory()
    elif model_name == 'shufflenet1_5':
        model = ShuffleNet1_5ArchitectureFactory()
    elif model_name == 'shufflenet2_0':
        model = ShuffleNet2_0ArchitectureFactory()
    elif model_name == 'vgg11bn':
        model = VGG11NetArchitectureFactory()
    elif model_name == 'vgg13bn':
        model = VGG13NetArchitectureFactory()
    elif model_name == 'vgg16bn':
        model = VGG16NetArchitectureFactory()
    elif model_name == 'vgg19bn':
        model = VGG19NetArchitectureFactory()

    return model
