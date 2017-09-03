import Jimp = require("jimp");
import Promise from "ts-promise";
const synaptic = require("synaptic");
const _ = require("lodash");

const originalImage = "original/captain.jpg";
const filterImage = "filters/whimsical.jpg";
const testImage = "test.jpg";

// Reading library parameters for the ANN simulation
const Neuron = synaptic.Neuron,
    Layer = synaptic.Layer,
    Network = synaptic.Network,
    Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

// Function to processs image and get RGB values: original image(original/captain.jpg)
function getImgData(fileName: string) {
    return new Promise((resolve, reject) => {
        Jimp.read(fileName).then((image) => {
            let inputSet: any = [];
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
                var red = image.bitmap.data[idx + 0];
                var green = image.bitmap.data[idx + 1];
                var blue = image.bitmap.data[idx + 2];
                var alpha = image.bitmap.data[idx + 3];
                
                // push data to the array
                inputSet.push([red, green, blue, alpha]);
            });
            // resolve input set in the promise
            resolve(inputSet);
        }).catch(function (err) {
            resolve([]);
        });
    });
}

// Initialize the perceptron network (ANN)
// 4 inputs, 5 node hidden layer, 5 node hidden layer and 4 outputs
const myPerceptron = new Architect.Perceptron(4, 5, 5, 4);
const trainer = new Trainer(myPerceptron);
const trainingSet: any = [];

getImgData(originalImage).then((inputs: any) => {
    getImgData(filterImage).then((outputs: any) => {
        // we have inputs and outputs
        for (let i = 0; i < inputs.length; i++) {
            // create the training set using normalized RGB values
            trainingSet.push({
                input: _.map(inputs[i], (val: any) => val / 255),
                output: _.map(outputs[i], (val: any) => val / 255)
            });
        }

        // start training
        trainer.train(trainingSet, {
            rate: .1, // training rate ()
            iterations: 200, // number of iterations
            error: .005, // termination error (This won't be achieved before 200 iterations)
            shuffle: true, // shuffle training data set
            log: 10, // logging frequency
            cost: Trainer.cost.CROSS_ENTROPY
        });
        
        Jimp.read(testImage).then((image) => {
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                var red = image.bitmap.data[idx + 0];
                var green = image.bitmap.data[idx + 1];
                var blue = image.bitmap.data[idx + 2];
                var alpha = image.bitmap.data[idx + 3];

                // Normalize and feed data to get the prediction
                var out = myPerceptron.activate([red / 255, green / 255, blue / 255, alpha / 255]);

                // Update red values by using the ANN
                image.bitmap.data[idx + 0] = _.round(out[0] * 255);
                // // Update green values by using the ANN
                image.bitmap.data[idx + 1] = _.round(out[1] * 255);
                // // // Update blue values by using the ANN
                image.bitmap.data[idx + 2] = _.round(out[2] * 255);
            });
            console.log('Writing output to output/'+testImage);
            image.write('output/' + testImage);
        }).catch(function (err) {
            console.error(err);
        });
    });
});