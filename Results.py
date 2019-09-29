from PrepImage import PrepImage
import matplotlib.pyplot as plt


class Results:
	@staticmethod
	def show(predictions, images, id_to_label_dict):
		count = 0
		max = len(images)
		for i in range(1, max + 1):
			plt.subplot(max, 1,i)
			plt.imshow(images[count])
			plt.title(id_to_label_dict[predictions[count]])
			count += 1
		plt.show()

def main():
	test_images = PrepImage.get_imgs("./input/test")
	Results.show([i for i in range(len(test_images))], test_images)

if __name__ == "__main__":
	main()
