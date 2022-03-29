import 'dotenv/config';
import express, { RequestHandler } from 'express';
import { Server, Socket } from 'socket.io';
import axios from 'axios';
import cors from 'cors';

import * as tf from "@tensorflow/tfjs";
import { nextFrame } from "@tensorflow/tfjs";


const app = express();
const io = new Server(3001);

// CORS
app.use((req, res, next) => {
	const origin = req.get('origin');
	const allowedOrigins = ['https://asclepius.krissada.com', 'http://localhost:3000'];
	if (allowedOrigins.includes(origin)) {
		cors({ origin: origin })(req, res, next);
	} else {
		cors({ origin: '*' })(req, res, next);
	}
});

// Check Content-Type
app.use((req, res, next) => {
	if (req.method.toUpperCase() == 'POST' && !req.is('application/json')) {
		res.status(415).send('only accepts application/json in POST method');
	} else {
		express.json()(req, res, next);
	}
	return;
});

app.use((err, req, res, next) => {
	if (err) {
		res.status(400).send('received malformed JSON');
	}
});

// let net: tf.GraphModel;

// tf.loadGraphModel(
// 	"https://github.com/Retaehc-pop/TF_SignLanguage_handvariation/raw/master/tfjs/group1-shard1of1.bin"
// ).then(model => net = model);

// tf.loadLayersModel('https://github.com/Retaehc-pop/TF_SignLanguage_handvariation/raw/master/tfjs/model.json');
// net.pre;
// const example = tf.frompixel(webcamElement);  // for example
// const prediction = model.predict(example);

// app.post('/detect', (req, res) => {
// 	const { image } = req.body;
// 	const tensor = tf.browser.fromPixels(image);
// 	const resized = tf.image.resizeBilinear(tensor, [224, 224]).toFloat();
// 	const offset = tf.scalar(127.5);
// 	const normalized = resized.sub(offset).div(offset);
// 	const batched = normalized.expandDims(0);
// 	const result = net.predict(batched);
// 	const prediction = result.dataSync();
// 	const predictionArray = Array.from(prediction);
// 	const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));
// 	const maxValue = predictionArray[maxIndex];
// 	const maxLabel = maxIndex.toString();
// 	const maxProbability = maxValue.toFixed(4);
// 	const response = {
// 		label: maxLabel,
// 		probability: maxProbability
// 	};
// 	res.send(response);
// });


app.listen(process.env.PORT || 3000, () => {
	console.log(`Listening on port ${process.env.PORT || 3000}`);
});

io.on('connection', socket => {
	socket.emit('hello', 'world');
	socket.on('hand', arg => {
		console.log(arg);
	});
});