import { loadLayersModel, LayersModel, Rank, Tensor } from '@tensorflow/tfjs';
import axios from 'axios';

let holisModel: LayersModel;
let handModel: LayersModel;
let numberModel: LayersModel;
let illnessModel: LayersModel;
let wordList = [];
let illnessList = [];

const threshold = 0.95;


// def extract_keypoint(result):
//   if result.multi_hand_landmarks:
//     for hand_landmarks in result.multi_hand_landmarks:
//       all_hand_pos = []
//       for joint in mp_hands.HandLandmark:
//         all_hand_pos.append(np.array([hand_landmarks.landmark[joint].x,hand_landmarks.landmark[joint].y,hand_landmarks.landmark[joint].z]))
//   return np.concatenate(all_hand_pos) # return 63 point of connection

export async function loadModel() {
	holisModel = await loadLayersModel('https://raw.githubusercontent.com/Retaehc-pop/TF_SignLanguage/master/hand/model.json');
	handModel = await loadLayersModel('https://raw.githubusercontent.com/Retaehc-pop/TF_SignLanguage_handvariation/master/tfjs/model.json');
	numberModel = await loadLayersModel('https://raw.githubusercontent.com/Retaehc-pop/TF_Signlanguage_number/master/number/model.json');
	illnessModel = await loadLayersModel('https://raw.githubusercontent.com/Retaehc-pop/TF_Signlanguage_sickness/master/illness/model.json');
	wordList = (await axios.get('https://raw.githubusercontent.com/Retaehc-pop/TF_SignLanguage/master/label_map.txt')).data.split('\n').map(word => word.split(':')[0]);
	illnessList = (await axios.get('https://raw.githubusercontent.com/Retaehc-pop/TF_Signlanguage_sickness/master/label_map.txt')).data.split('\n').map(word => word.split(':')[0]);
	if (!holisModel) throw 'Holistic model not loaded';
	if (!handModel) throw 'Hand model not loaded';
	if (!numberModel) throw 'Number model not loaded';
	if (!illnessModel) throw 'Illness model not loaded';
}

export function predictWord(input: Tensor<Rank> | Tensor<Rank>[]) {
	let prediction: Float32Array;
	try {
		prediction = (<Tensor<Rank>>holisModel.predict(input)).dataSync() as Float32Array;
		console.log(prediction);
		const max = prediction.reduce((max, cur) => max = cur > max ? cur : max, 0);
		if (max < 0.6) return;
		return { word: wordList[prediction.indexOf(max)], confidence: max };
	} catch (err) {
		console.warn(`Warning: ${err}`);
		return;
	}
}

export function predictIllness(input: Tensor<Rank> | Tensor<Rank>[]) {
	let prediction: Float32Array;
	try {
		prediction = (<Tensor<Rank>>illnessModel.predict(input)).dataSync() as Float32Array;
		console.log(prediction);
		const max = prediction.reduce((max, cur) => max = cur > max ? cur : max, 0);
		if (max < 0) return;
		return { word: illnessList[prediction.indexOf(max)], confidence: max };
	} catch (err) {
		console.log('gonna warn');
		console.warn(`Warning: ${err}`);
		return;
	}
}

export function predictLetter(input: Tensor<Rank> | Tensor<Rank>[]) {
	let prediction: Float32Array;
	try {
		prediction = (<Tensor<Rank>>handModel.predict(input)).dataSync() as Float32Array;
		const max = prediction.reduce((max, cur) => max = cur > max ? cur : max, 0);
		// console.log(max);
		if (max < threshold) return;
		return { letter: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'][prediction.indexOf(max)], confidence: max };
	} catch (err) {
		console.warn(`Warning: ${err}`);
		return;
	}
}

export function predictNumber(input: Tensor<Rank> | Tensor<Rank>[]) {
	let prediction: Float32Array;
	try {
		prediction = (<Tensor<Rank>>numberModel.predict(input)).dataSync() as Float32Array;
		const max = prediction.reduce((max, cur) => max = cur > max ? cur : max, 0);
		if (max < threshold) return;
		return { letter: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][prediction.indexOf(max)], confidence: max };
	} catch (err) {
		console.warn(`Warning: ${err}`);
		return;
	}
}