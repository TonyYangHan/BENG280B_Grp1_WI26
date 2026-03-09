import os

import matplotlib.pyplot as plt
import torch


def dice_coefficient(pred, target, smooth: float = 1.0):
	pred = (pred > 0.5).float()
	intersection = (pred * target).sum()
	return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def visualize_predictions(model, dataloader, device: torch.device, save_dir: str, num_samples: int = 6):
	"""Visualize predictions on lesion-positive slices and save a grid image."""
	model.eval()

	samples_collected = 0
	fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

	with torch.no_grad():
		for batch in dataloader:
			if samples_collected >= num_samples:
				break

			images = batch['image'].to(device)
			masks = batch['mask'].to(device)
			has_lesion = batch['has_lesion']

			lesion_indices = [i for i, hl in enumerate(has_lesion) if hl]
			if not lesion_indices:
				continue

			logits, _ = model(images)
			probs = torch.sigmoid(logits)

			for idx in lesion_indices:
				if samples_collected >= num_samples:
					break

				image = images[idx].cpu()
				mask = masks[idx].cpu()
				pred = (probs[idx, 0].cpu() > 0.5).float()

				row = samples_collected

				axes[row, 0].imshow(image[0], cmap='gray')
				axes[row, 0].set_title('Input (DWI)')
				axes[row, 0].axis('off')

				if image.shape[0] > 1:
					axes[row, 1].imshow(image[1], cmap='gray')
					axes[row, 1].set_title('Input (ADC)')
				else:
					axes[row, 1].imshow(image[0], cmap='gray')
					axes[row, 1].set_title('Input')
				axes[row, 1].axis('off')

				axes[row, 2].imshow(mask, cmap='Reds', vmin=0, vmax=1)
				axes[row, 2].set_title('Ground Truth')
				axes[row, 2].axis('off')

				axes[row, 3].imshow(pred, cmap='Reds', vmin=0, vmax=1)
				dice = dice_coefficient(pred.unsqueeze(0), mask.unsqueeze(0))
				axes[row, 3].set_title(f'Prediction (Dice: {dice:.3f})')
				axes[row, 3].axis('off')

				samples_collected += 1

	plt.tight_layout()
	os.makedirs(save_dir, exist_ok=True)
	plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150, bbox_inches='tight')
	plt.close()

	print(f"✓ Saved visualization to {save_dir}/predictions.png")
