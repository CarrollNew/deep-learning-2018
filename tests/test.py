from resnext import *

def test_resnext():
    train_dataset = dsets.FashionMNIST(root='downloaded_1',
                                        train=True,
                                        transform = transforms.Compose([transforms.Resize(224),
                                                                        transforms.Grayscale(3),
                                                                        transforms.ToTensor()]),
                                        download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True);

    labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
                      7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
    model = resnext()
    trainer = Trainer(model, train_loader, epochs=1, learning_rate=0.01, output_dir='.')
    trainer.train_model()
