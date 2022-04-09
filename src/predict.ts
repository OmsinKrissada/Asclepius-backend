import { loadLayersModel, LayersModel, Rank, Tensor } from '@tensorflow/tfjs';
import axios from 'axios';

let holisModel: LayersModel;
let handModel: LayersModel;
let wordList = [];

const threshold = 0.95;


// def extract_keypoint(result):
//   if result.multi_hand_landmarks:
//     for hand_landmarks in result.multi_hand_landmarks:
//       all_hand_pos = []
//       for joint in mp_hands.HandLandmark:
//         all_hand_pos.append(np.array([hand_landmarks.landmark[joint].x,hand_landmarks.landmark[joint].y,hand_landmarks.landmark[joint].z]))
//   return np.concatenate(all_hand_pos) # return 63 point of connection

export async function loadModel() {
	holisModel = await loadLayersModel('https://raw.githubusercontent.com/Retaehc-pop/TF_SignLanguage/master/tfjs/model.json');
	handModel = await loadLayersModel('https://raw.githubusercontent.com/Retaehc-pop/TF_SignLanguage_handvariation/master/tfjs/model.json');
	wordList = (await axios.get('https://raw.githubusercontent.com/Retaehc-pop/TF_SignLanguage/master/word_list.txt')).data.split('\n');
	if (!holisModel) throw 'Holistic model not loaded';
	if (!handModel) throw 'Hand model not loaded';
}

export function predictWord(input: Tensor<Rank> | Tensor<Rank>[]) {
	let prediction: Float32Array;
	try {
		prediction = (<Tensor<Rank>>holisModel.predict(input)).dataSync() as Float32Array;
		console.log(prediction);
		const max = prediction.reduce((max, cur) => max = cur > max ? cur : max, 0);
		if (max < threshold) return;
		return { word: wordList[prediction.indexOf(max)], confidence: max };
	} catch (err) {
		console.warn(`Warning: ${err}`);
		return;
	}
}
export function predictLetter(input: Tensor<Rank> | Tensor<Rank>[]) {
	let prediction: Float32Array;
	try {
		prediction = (<Tensor<Rank>>handModel.predict(input)).dataSync() as Float32Array;
		console.log(prediction);
		const max = prediction.reduce((max, cur) => max = cur > max ? cur : max, 0);
		// console.log(max);
		if (max < threshold) return;
		return { letter: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'][prediction.indexOf(max)], confidence: max };
	} catch (err) {
		console.warn(`Warning: ${err}`);
		return;
	}
}


