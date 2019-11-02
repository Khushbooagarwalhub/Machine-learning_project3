from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import pandas as pd

# Function to build and train fine-tuned models
def build_train_model(model, train_generator, test_generator, model_name):
    my_model = add_layers_to_model(model)
    history = compile_train_model(my_model, train_generator, test_generator)
    save_record_of_history(history, model_name)

# Function to add dense fully-connected layers and 
# output layers to pre-trained models
def add_layers_to_model(model):
    new_block = model.output

    new_block = Flatten()(new_block)
    new_block = Dense(512, activation="selu")(new_block)
    new_block = BatchNormalization()(new_block)
    new_block = Dropout(0.5)(new_block)
    new_block = Dense(128, activation="selu", name="features")(new_block)
    new_block = BatchNormalization()(new_block)
    new_block = Dropout(0.5)(new_block)
    new_block = Dense(10, activation="softmax")(new_block)
    
    new_model = Model(inputs=model.input, outputs=new_block)
    
    return new_model

# Function to compile and train new model
def compile_train_model(model, train_generator, test_generator):
    model.compile(optimizer=Adam(lr=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"]
             )
    
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = len(train_generator),
                                  validation_data = test_generator,
                                  validation_steps = len(test_generator),
                                  epochs= 70
                                )
    return history

# Function to save training history to csv file
def save_record_of_history(history, model_name):
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']

    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    file = f"../Output/fine_tune_history/ten/{model_name}.csv"
    hist = {
        "train loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    }
    df = pd.DataFrame(hist)
    df.to_csv(file)

