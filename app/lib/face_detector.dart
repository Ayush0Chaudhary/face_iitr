
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'dart:io';
import 'dart:async';

List<CameraDescription> cameras = [];

// Future<void> main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   cameras = await availableCameras();
//   runApp(MyApp());
// }

// class Face extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(home: FaceDetectionCamera());
//   }
// }

class FaceDetectionCamera extends StatefulWidget {
  @override
  _FaceDetectionCameraState createState() => _FaceDetectionCameraState();
}

class _FaceDetectionCameraState extends State<FaceDetectionCamera> {
  late CameraController _cameraController;
  late FaceDetector _faceDetector;
  bool _isPictureTaken = false;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableLandmarks: true,
        enableClassification: true,
        enableTracking: true,
      ),
    );
  }

  void _initCamera() async {
    _cameraController = CameraController(
      cameras[1],
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _cameraController.initialize();

    setState(() {});

    _timer = Timer.periodic(Duration(seconds: 2), (timer) {
      if (!_isPictureTaken) {
        _captureAndDetectFace();
      }
    });
  }

  void _captureAndDetectFace() async {
    if (!_cameraController.value.isInitialized || _isPictureTaken) return;

    _isPictureTaken = true;

    try {
      final file = await _cameraController.takePicture();
      final inputImage = InputImage.fromFilePath(file.path);
      final List<Face> faces = await _faceDetector.processImage(inputImage);

      if (faces.isNotEmpty) {
        final Face face = faces.first;
        final leftEye = face.landmarks[FaceLandmarkType.leftEye];
        final rightEye = face.landmarks[FaceLandmarkType.rightEye];
        
        if (leftEye != null && rightEye != null) {
          print(leftEye.position);
        print(rightEye.position);
          await Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => DisplayPictureScreen(imagePath: file.path),
            ),
          );
        } else {
          debugPrint('Both eyes not visible.');
        }
      } else {
        debugPrint('No faces detected');
      }
    } catch (e) {
      debugPrint('Error: $e');
    }

    _isPictureTaken = false;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _cameraController.dispose();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: EdgeInsets.symmetric(vertical: 10, horizontal: 20),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  'Ensure both eyes are visible',
                  style: TextStyle(color: Colors.white, fontSize: 16),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
class DisplayPictureScreen extends StatefulWidget {
  final String imagePath;

  const DisplayPictureScreen({required this.imagePath});

  @override
  State<DisplayPictureScreen> createState() => _DisplayPictureScreenState();
}

class _DisplayPictureScreenState extends State<DisplayPictureScreen> {
  Face? _face;
  late FaceDetector _faceDetector;
  late Size _imageSize;

  @override
  void initState() {
    super.initState();
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(enableLandmarks: true),
    );
    _detectFace();
  }

  void _detectFace() async {
    final inputImage = InputImage.fromFilePath(widget.imagePath);
    final decodedImage = await decodeImageFromList(File(widget.imagePath).readAsBytesSync());

    _imageSize = Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());

    final List<Face> faces = await _faceDetector.processImage(inputImage);
    if (faces.isNotEmpty) {
      setState(() {
        _face = faces.first;
      });
    }
  }

  @override
  void dispose() {
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Captured Face')),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                Image.file(File(widget.imagePath)),
                if (_face != null)
                  CustomPaint(
                    painter: LandmarkPainter(
                      face: _face!,
                      imageSize: _imageSize,
                    ),
                    size: _imageSize,
                  ),
              ],
            ),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
            },
            child: Text("Back to Camera"),
          ),
        ],
      ),
    );
  }
}

class LandmarkPainter extends CustomPainter {
  final Face face;
  final Size imageSize;

  LandmarkPainter({required this.face, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.fill;

    final textStyle = TextStyle(color: Colors.white, fontSize: 12);

    final drawLabel = (String label, Offset offset) {
      final textSpan = TextSpan(text: label, style: textStyle);
      final tp = TextPainter(text: textSpan, textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, offset);
    };

    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    face.landmarks.forEach((type, landmark) {
      final pos = Offset(
        landmark!.position.x * scaleX,
        landmark.position.y * scaleY,
      );

      canvas.drawCircle(pos, 4, paint);
      drawLabel(_landmarkLabel(type), pos + Offset(5, -5));
    });
  }

  String _landmarkLabel(FaceLandmarkType type) {
    switch (type) {
      case FaceLandmarkType.leftEye:
        return "Left Eye";
      case FaceLandmarkType.rightEye:
        return "Right Eye";
      case FaceLandmarkType.leftEar:
        return "Left Ear";
      case FaceLandmarkType.rightEar:
        return "Right Ear";
      case FaceLandmarkType.leftCheek:
        return "Left Cheek";
      case FaceLandmarkType.rightCheek:
        return "Right Cheek";
      case FaceLandmarkType.noseBase:
        return "Nose Base";
      case FaceLandmarkType.leftMouth:
        return "Mouth Left";
      case FaceLandmarkType.rightMouth:
        return "Mouth Right";
      case FaceLandmarkType.bottomMouth:
        return "Mouth Bottom";
      default:
        return "Unknown";
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
