1. optimizer = adam (learning rate 1e-4)
2. loss function = perceptual loss (function for this:
# Perceptual loss using VGG16 features
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(480, 640, 3))  # Update input shape
vgg.trainable = False
output_layer = vgg.get_layer('block3_conv3').output
vgg_model = Model(vgg.input, output_layer)

def perceptual_loss(y_true, y_pred):
    y_true_features = vgg_model(y_true)
    y_pred_features = vgg_model(y_pred)
    return mean_squared_error(y_true_features, y_pred_features)
)
	loss value of only 1000 epoch model available
3. loss value (after 1000 epochs, batch size 32) = 0.2232
	val_loss = 0.5056