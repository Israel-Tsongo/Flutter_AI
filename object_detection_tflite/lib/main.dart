import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:object_detection_tflite/ui/home_view.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {

  const MyApp ({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detection TFLite',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomeView(),
    );
  }
}
