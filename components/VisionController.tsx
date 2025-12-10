import React, { useEffect, useRef } from 'react';
import { FilesetResolver, Hands, HandLandmarker, HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { HandGesture, HandState } from '../types';

interface VisionControllerProps {
  onUpdate: (state: HandState) => void;
}

export const VisionController: React.FC<VisionControllerProps> = ({ onUpdate }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const lastVideoTime = useRef(-1);
  const requestRef = useRef<number>(0);
  const landmarkerRef = useRef<HandLandmarker | null>(null);

  useEffect(() => {
    let active = true;

    const setup = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        if (!active) return;

        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        
        landmarkerRef.current = landmarker;
        startCamera();
      } catch (err) {
        console.error("Error initializing vision:", err);
      }
    };

    setup();

    return () => {
      active = false;
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(t => t.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener('loadeddata', predictWebcam);
      }
    } catch (err) {
      console.error("Camera denied:", err);
    }
  };

  const predictWebcam = () => {
    if (!landmarkerRef.current || !videoRef.current) return;

    let startTimeMs = performance.now();
    
    if (lastVideoTime.current !== videoRef.current.currentTime) {
      lastVideoTime.current = videoRef.current.currentTime;
      const result = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
      processLandmarks(result);
    }
    
    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  const processLandmarks = (result: HandLandmarkerResult) => {
    if (result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      
      // 1. Detect Gesture (Fist vs Open)
      // Check average distance of finger tips to wrist
      const wrist = landmarks[0];
      const tips = [8, 12, 16, 20].map(i => landmarks[i]); // Index, Middle, Ring, Pinky tips
      
      // Calculate distances from wrist
      const dists = tips.map(tip => Math.sqrt(
        Math.pow(tip.x - wrist.x, 2) + 
        Math.pow(tip.y - wrist.y, 2) + 
        Math.pow(tip.z - wrist.z, 2)
      ));
      
      const avgDist = dists.reduce((a, b) => a + b, 0) / dists.length;
      
      // Thumb tip to Index tip for Pinch
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];
      const pinchDist = Math.sqrt(
        Math.pow(thumbTip.x - indexTip.x, 2) +
        Math.pow(thumbTip.y - indexTip.y, 2)
      );

      let gesture = HandGesture.OPEN_PALM;
      if (avgDist < 0.25) { // Threshold for fist roughly
        gesture = HandGesture.CLOSED_FIST;
      }
      
      // 2. Rotation Calculation (Map X coordinate of wrist to rotation)
      // Normalizing x from [0, 1] to [-PI, PI]
      const rotX = (wrist.x - 0.5) * 4; // Multiplier for sensitivity
      const rotY = (wrist.y - 0.5) * 2;

      onUpdate({
        gesture,
        rotation: { x: rotX, y: rotY },
        pinchDistance: pinchDist,
        isPresent: true
      });

    } else {
      onUpdate({
        gesture: HandGesture.NONE,
        rotation: { x: 0, y: 0 },
        pinchDistance: 1,
        isPresent: false
      });
    }
  };

  return (
    <video 
      ref={videoRef} 
      autoPlay 
      playsInline 
      className="fixed bottom-4 right-4 w-32 h-24 object-cover rounded-lg border-2 border-green-800 opacity-50 pointer-events-none z-50 hidden md:block"
      muted
    />
  );
};
