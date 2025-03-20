import { createApp } from 'vue';
import App from './App.vue';
import DiabetesDetector from './DiabetesDetector.vue';

const app = createApp(DiabetesDetector);
app.component('diabetes-detector', DiabetesDetector);
app.mount('#app');
