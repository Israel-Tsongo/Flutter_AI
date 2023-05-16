
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:object_detection_tflite/tflite/classifier.dart';
import 'package:object_detection_tflite/tflite/recognition.dart';
import 'package:object_detection_tflite/tflite/stats.dart';
import 'package:object_detection_tflite/ui/camera_view_singleton.dart';
import 'package:object_detection_tflite/utils/isolate_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// [CameraView] sends each frame for inference
class CameraView extends StatefulWidget {
  /// Callback to pass results after inference to [HomeView]
  final Function(List<Recognition> recognitions) resultsCallback;

  /// Callback to inference stats to [HomeView]
  final Function(Stats stats) statsCallback;

  /// Constructor
  const CameraView( this.resultsCallback, this.statsCallback);

  @override
  State createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> with WidgetsBindingObserver {
  /// List of available cameras
  List<CameraDescription>? cameras;

  /// Controller
  CameraController? cameraController;

  /// true when inference is ongoing
  bool? predicting;

  /// Instance of [Classifier]
  Classifier? classifier;

  /// Instance of Interpreter
  Interpreter? interpreter;

  /// Labels file loaded as list
  List<String>? labels;

  /// Instance of [IsolateUtils]
  IsolateUtils? isolateUtils;

  @override
  void initState() {
    super.initState();
    initStateAsync();
  }

  void initStateAsync() async {
    WidgetsBinding.instance.addObserver(this);

    // Spawn a new isolate
    isolateUtils = IsolateUtils();
    await isolateUtils!.start();

    // Create an instance of classifier to load model and labels
    classifier =  Classifier();//initialization not for prediction

    interpreter= await classifier!.loadModel(interpreter: null);
    labels=await classifier!.loadLabels(labels: null);

    print("==initStateAsync()==>classifierobj : $classifier");
    print("==inter: $interpreter");
    print("==labels : $labels");

    // Camera initialization
    initializeCamera();

    // Initially predicting = false
    predicting = false;
  }

  /// Initializes the camera by setting [cameraController]
  void initializeCamera() async {
    cameras = await availableCameras();

    // cameras[0] for rear-camera
    cameraController =
      CameraController(cameras![0], ResolutionPreset.low, enableAudio: false);

      cameraController!.initialize().then((_) async {
      // Stream of image passed to [onLatestImageAvailable] callback

      await cameraController!.startImageStream(onLatestImageAvailable);

      /// previewSize is size of each image frame captured by controller
      ///
      /// 352x288 on iOS, 240p (320x240) on Android with ResolutionPreset.low
      Size previewSize = cameraController!.value.previewSize!;

      /// previewSize is size of raw input image to the model
      CameraViewSingleton.inputImageSize = previewSize;

      // the display width of image on screen is
      // same as screenWidth while maintaining the aspectRatio
      Size screenSize = MediaQuery.of(context).size;
      CameraViewSingleton.screenSize = screenSize;
      CameraViewSingleton.ratio = screenSize.width / previewSize.height;
    });
  }

  @override
  Widget build(BuildContext context) {
    // Return empty container while the camera is not initialized


    if (cameraController == null || !cameraController!.value.isInitialized) {
      print("========>No camera<=========");
      return Container();
    }
    print("Camera activated");

    return AspectRatio(
        aspectRatio: cameraController!.value.aspectRatio,
        child: CameraPreview(cameraController!));
  }

  /// Callback to receive each frame [CameraImage] perform inference on it
  onLatestImageAvailable(CameraImage cameraImage) async {
    print("===onLatestImageAvailable====>classifierobj : $classifier");
    print("====>interpreter : ${interpreter!.address}");
    print("====>labels : $labels");

    if (interpreter != null && labels != null) {
      // If previous inference has not completed then return

      if(predicting ==null) {

        return;
      }

      setState(() {
        predicting = true;
      });

      var uiThreadTimeStart = DateTime.now().millisecondsSinceEpoch;

      // Data to be passed to inference isolate
      var isolateData = IsolateData(
          cameraImage, interpreter!.address, labels);

      print("isolateData is:label-> ${isolateData.labels}, interpreterAddress-> ${isolateData.interpreterAddress}");

      // We could have simply used the compute method as well however
      // it would be as in-efficient as we need to continuously passing data
      // to another isolate.

      /// perform inference in separate isolate
      Map<String, dynamic> inferenceResults = await inference(isolateData);

      var uiThreadInferenceElapsedTime =
          DateTime.now().millisecondsSinceEpoch - uiThreadTimeStart;

      // pass results to HomeView
      widget.resultsCallback(inferenceResults["recognitions"]);

      // pass stats to HomeView
      widget.statsCallback((inferenceResults["stats"] as Stats) ..totalElapsedTime = uiThreadInferenceElapsedTime);

      // set predicting to false to allow new frames
      setState(() {
        predicting = false;
      });
    }
  }

  /// Runs inference in another isolate
  Future<Map<String, dynamic>> inference(IsolateData isolateData) async {
    ReceivePort responsePort = ReceivePort();

    isolateUtils!.sendPort! // simulation message :voici le sendPort[responsePort.sendPort] sur le quel tu va m'envoyer le resurltat
        .send(isolateData..responsePort = responsePort.sendPort);//sending isolateData to the [final ReceivePort _receivePort = ReceivePort() in isolate_utils file]

    var results = await responsePort.first;//return the first returned message isolate_utils

    print("The predicted result From cameraview file: $results");

    return results;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async {
    switch (state) {
      case AppLifecycleState.paused:
        cameraController!.stopImageStream();
        break;
      case AppLifecycleState.resumed:
        if (!cameraController!.value.isStreamingImages) {
          await cameraController!.startImageStream(onLatestImageAvailable);
        }
        break;
      default:
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    cameraController!.dispose();
    super.dispose();
  }
}
