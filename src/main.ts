require('dotenv').config();
import { createServer } from 'http';
import { Server } from 'socket.io';
import { expandDims } from "@tensorflow/tfjs";
import { NormalizedLandmarkList, NormalizedLandmarkListList } from '@mediapipe/hands';

import { loadModel, predictLetter, predictWord } from './predict';

console.log('Starting ...');

const ws_port = process.env.PORT || 3001;
const wsHttpServer = createServer();
const io = new Server(wsHttpServer, {
	cors: {
		origin: "*"
	}, path: '/'
});

interface holisResult {
	pose: NormalizedLandmarkList;
	face: NormalizedLandmarkList;
	left: NormalizedLandmarkList;
	right: NormalizedLandmarkList;
}

const sessionConfident = new Map<string, string[]>();
const lastPrediction = new Map<string, string>();
const frameCoordinates = new Map<string, number[][]>(); // store last 30 frame results from Holistic model

io.on('connection', async socket => {
	console.log(`${new Date().toLocaleTimeString('th', { hour12: false })} [CONNECTED] ${socket.id} (${(await io.allSockets()).size} connected) IP: ${socket.handshake.address}`);
	sessionConfident.set(socket.id, []);
	frameCoordinates.set(socket.id, []);
	socket.on('disconnect', async reason => {
		console.log(`${new Date().toLocaleTimeString('th', { hour12: false })} [DISCONNECTED] ${socket.id} (${(await io.allSockets()).size} connected) Reason: ${reason}`);
		sessionConfident.delete(socket.id);
		lastPrediction.delete(socket.id);
		frameCoordinates.delete(socket.id);
	});

	// socket.on('frame', (frame: HTMLVideoElement) => {
	// 	console.log(frame.width);
	// });

	socket.on('holis', (result: holisResult) => {
		// console.log('received');
		// Format data
		const combined = [];
		// console.log(result.pose.length, result.face.length, result.left.length, result.right.length);
		for (const landmark of result.pose) {
			combined.push(landmark.x);
			combined.push(landmark.y);
			combined.push(landmark.z);
			combined.push(landmark.visibility);
		}
		for (const landmark of result.face) {
			combined.push(landmark.x);
			combined.push(landmark.y);
			combined.push(landmark.z);
		}
		for (const landmark of result.left) {
			combined.push(landmark.x);
			combined.push(landmark.y);
			combined.push(landmark.z);
		}
		for (const landmark of result.right) {
			combined.push(landmark.x);
			combined.push(landmark.y);
			combined.push(landmark.z);
		}

		// Predict
		let frames = frameCoordinates.get(socket.id);
		frames.push(combined);
		frames = frames.slice(-30);
		frameCoordinates.set(socket.id, frames);
		if (frames.length < 30) return;

		const prediction = predictWord(expandDims(frames));
		console.log(prediction);
		if (!prediction) return;
		const { word, confidence } = prediction;

		socket.emit('word', { word, confidence });

	});


	socket.on('mh', (landmarks_list: NormalizedLandmarkListList) => {
		// Format data
		const all_hand_pos = [];
		for (const landmarks of landmarks_list) {
			for (const joint of landmarks) {
				all_hand_pos.push(joint.x);
				all_hand_pos.push(joint.y);
				all_hand_pos.push(joint.z);
			}
		}

		// Predict
		const prediction = predictLetter(expandDims(all_hand_pos));
		if (!prediction) return;
		socket.emit('letter', prediction);
	});
});

loadModel().then(() => {
	wsHttpServer.listen(ws_port, () => console.log('Listening for socket.io on port ' + ws_port));
});