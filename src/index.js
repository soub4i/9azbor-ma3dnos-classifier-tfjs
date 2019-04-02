import "./styles.css";

import * as tf from '@tensorflow/tfjs';


function loadMobilenet() {
  return tf.loadModel('./model/ml-classifier-parsley-coriander.json');
}


document.getElementById("app").innerHTML = `<h1>Hello</h1>`;

