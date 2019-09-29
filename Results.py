from PrepImage import PrepImage
import matplotlib.pyplot as plt


class Results:
	@staticmethod
	def show(images, predictions=None, id_to_label=None, imgs_per_row=4):
		n, count = len(images), 0

		row = int(n/imgs_per_row) 
		row = row if (n % imgs_per_row == 0) else row + 1	# division overflow
		col = int(n/row) 
		col = col if (n % row == 0) else col + 1	# division overflow
		# print("n={}, imgs_per_row={}, row={}, col={}".format(n, imgs_per_row, row, col))

		fig, ax = plt.subplots(row, col)
		fig.suptitle("Prediction Results on Test Set")

		for i in range(len(ax)):	# row
			for j in range(len(ax[i])):	# col
				axx = ax[i][j]
				axx.imshow(images[count])

				# Add title
				if id_to_label is not None and predictions is not None:
					name = id_to_label[predictions[count]]
				else:
					name = "i={}".format(count)
				axx.set_title(name)

				# Hide axis labels
				axx.get_xaxis().set_visible(False)
				axx.get_yaxis().set_visible(False)
				axx.get_xaxis().set_ticks([])
				axx.get_yaxis().set_ticks([])

				count += 1
				if count == n:
					break
			if count == n:
				break
		plt.show()


# Testing Purposes
def testRun():
	test_images = PrepImage.get_imgs("./input/test")
	Results.show(test_images[1:9], predictions=None, id_to_label=None)

if __name__ == "__main__":
	testRun()
