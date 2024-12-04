import torch

def evaluate(model, data_loader):
    DEVICE = torch.device('mps')
    model.eval()
    model.to(DEVICE)

    source_loader = data_loader['MNIST']
    target_dataset = [key for key in data_loader.keys() if key != 'MNIST'][0]
    target_loader = data_loader[target_dataset]

    domain_correct, domain_total = 0, 0
    source_label_correct, source_label_total = 0, 0
    target_label_correct, target_label_total = 0, 0

    with torch.no_grad():
        for data, labels in source_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            _, _, _, _, _, _, label_logits, _ = model(data, labels)
            _, predicted = torch.max(label_logits, 1)
            source_label_total += labels.size(0)
            source_label_correct += (predicted == labels).sum().item()

        print(f"Y-acc MNIST: {100 * source_label_correct / source_label_total:.2f}%", end = ' | ')
        
        for data, labels in target_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            _, _, _, _, _, _, label_logits, _ = model(data, labels)
            _, predicted = torch.max(label_logits, 1)
            target_label_total += labels.size(0)
            target_label_correct += (predicted == labels).sum().item()

        print(f"Y-acc {target_dataset}: {100 * target_label_correct / target_label_total:.2f}%", end=' | ')

        for domain, loader in enumerate([source_loader, target_loader]):
            for data, _ in loader:
                data = data.to(DEVICE)

                _, _, _, _, _, _, _, domain_logits = model(data, reverse_grad=True)
                _, predicted = torch.max(domain_logits, 1)

                domain_total += data.size(0)
                domain_correct += (predicted == domain).sum().item()

        print(f"D-acc: {100 * domain_correct / domain_total:.2f}%", end=' | ')