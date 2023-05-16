import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as image_lib;
import 'package:object_detection_tflite/tflite/classifier.dart';
import 'package:object_detection_tflite/utils/image_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Manages separate Isolate instance for inference
class IsolateUtils {
  static const String DEBUG_NAME = "InferenceIsolate";

  Isolate? _isolate;

  final ReceivePort _receivePort = ReceivePort();
  SendPort? _sendPort;

  SendPort? get sendPort => _sendPort;

  Future<void> start() async {
    _isolate = await Isolate.spawn<SendPort>(
      entryPoint,
      _receivePort.sendPort, //this is the message
      debugName: DEBUG_NAME,
    );

    _sendPort = await _receivePort.first;
  }

  static void entryPoint(SendPort sendPort) async {
    print("Inside entryPoint from Isolate file ");

    final port = ReceivePort();
    sendPort.send(port.sendPort); // Sending port.sendPort to _receivePort received by it lisner

    await for (final IsolateData isolateData in port) {

        Classifier classifier = Classifier(
            interpreter:
                Interpreter.fromAddress(isolateData.interpreterAddress!),
            labels: isolateData.labels);

        image_lib.Image image =
            ImageUtils.convertCameraImage(isolateData.cameraImage!)!;
        if (Platform.isAndroid) {
          image = image_lib.copyRotate(image, 90);
        }
        Map<String, dynamic> results = classifier.predict(image)!;// call for prediction
        isolateData.responsePort!.send(results); // Sending results to responsePort from cameraview received by it lesner (first)

        print("The image to predict : $image");
        print("The predicted result is: $results");
    }
  }
}

/// Bundles data to pass between Isolate
class IsolateData {
  CameraImage? cameraImage;
  int? interpreterAddress;
  List<String>? labels;
  SendPort? responsePort;

  IsolateData(
    this.cameraImage,
    this.interpreterAddress,
    this.labels,
  );
}
