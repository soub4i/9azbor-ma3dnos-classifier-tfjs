import * as tf from "@tensorflow/tfjs";
import "babel-polyfill";

const classes = ["معدنوس", "قزبر"];

const readURL = input => {
  return new Promise((resolve, reject) => {
    if (input.files && input.files[0]) {
      var reader = new FileReader();

      reader.onload = e => {
        document
          .querySelector("#display-img")
          .setAttribute("src", e.target.result);
        document.querySelector("#display-img").setAttribute("width", "150");
        document.querySelector("#display-img").setAttribute("height", "150");

        const image = new Image();
        image.src = e.target.result;
        image.crossOrigin = "Anonymous";

        image.onload = () => resolve(image);
        image.onerror = err => reject(err);
        resolve(reader);
      };
      reader.readAsDataURL(input.files[0]);
    }
  });
};

document
  .querySelector("#test-img")
  .addEventListener("change", async function(event) {
    document.querySelector("#desc").textContent = "";

    const meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);

    readURL(this)
      .then(async (img: any) => {
        const model = await tf.loadLayersModel(`model/model.json`);
        let imgElm = document.querySelector("#display-img");

        const tensor = tf.browser
          .fromPixels(imgElm)
          .resizeNearestNeighbor([150, 150])
          .toFloat()
          .sub(meanImageNetRGB)
          .reverse(2)
          .expandDims();
        const prediction = model.predict(tensor);

        let predictions = await model.predict(tensor).data();

        // get the model's prediction results
        console.log(Array.from(predictions));

        document.querySelector("#desc").textContent =
          classes[
            prediction
              .as1D()
              .argMax()
              .dataSync()[0]
          ];
      })
      .catch(err => console.log(err));
  });
