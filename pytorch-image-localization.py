
class ImageLocalizationBase(nn.Module):
    def training_step(self, batch):
        images, labels, bboxes = batch
        out_class, out_bbox = self(images) 
        loss_class = F.cross_entropy(out_class, labels)  
        loss_bbox = F.mse_loss(out_bbox, bboxes)        
        loss = loss_class + loss_bbox                   
        return loss

    def validation_step(self, batch):
        images, labels, bboxes = batch
        out_class, out_bbox = self(images)
        loss_class = F.cross_entropy(out_class, labels)
        loss_bbox = F.mse_loss(out_bbox, bboxes)
        loss = loss_class + loss_bbox
        
        acc = accuracy(out_class, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageLocalizationModel(ImageLocalizationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Flatten(),
            nn.Linear(256*8*8, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU()
        )
       
        self.classifier = nn.Linear(512, 10)  
        self.bbox_regressor = nn.Linear(512, 4)  

    def forward(self, xb):
        features = self.network(xb)
        out_class = self.classifier(features)  
        out_bbox = self.bbox_regressor(features)  #
        return out_class, out_bbox
