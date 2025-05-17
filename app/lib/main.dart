import 'dart:io';
import 'package:flutter/material.dart';
import 'package:glassmorphism_ui/glassmorphism_ui.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_spinkit/flutter_spinkit.dart';

void main() => runApp(faceRecognizerApp());

class faceRecognizerApp extends StatelessWidget {
  const faceRecognizerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'face ID',
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        scaffoldBackgroundColor: Colors.black,
      ),
      home: faceCaptureScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class faceCaptureScreen extends StatefulWidget {
  const faceCaptureScreen({super.key});

  @override
  _faceCaptureScreenState createState() => _faceCaptureScreenState();
}

class _faceCaptureScreenState extends State<faceCaptureScreen> {
  File? _image;
  String? _userId;
  bool _loading = false;
  TextEditingController userctrl = TextEditingController();
  TextEditingController ipController = TextEditingController();

  Future<void> _pickImage() async {
    final picked = await ImagePicker().pickImage(source: ImageSource.camera);
    if (picked != null) {
      setState(() {
        _image = File(picked.path);
        _userId = null;
      });
    }
  }

  Future<void> _sendImageForEnrollment() async {
    if (_image == null || ipController.text.isEmpty) return;

    setState(() => _loading = true);

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://${ipController.text}:8000/register'),
    );

    request.fields['userId'] = userctrl.text;
    userctrl.text = "";
    request.files.add(await http.MultipartFile.fromPath('file', _image!.path));
    try {
      final response = await request.send();
      final body = await response.stream.bytesToString();
      print("Enroll response: $body");

      setState(() {
        _userId =
            response.statusCode == 200 ? "Enrolled: $body" : "Error: $body";
      });
    } catch (e) {
      print("Error: $e");
      setState(() {
        _userId = "Enroll failed: $e";
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _sendImageForIdentification() async {
    if (_image == null || ipController.text.isEmpty) return;

    setState(() => _loading = true);

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://${ipController.text}:8000/identify'),
    );
    request.files.add(await http.MultipartFile.fromPath('file', _image!.path));
    try {
      final response = await request.send();
      final body = await response.stream.bytesToString();
      print(body);
      setState(() {
        _userId = response.statusCode == 200 ? body : "Error: $body";
      });
    } catch (e) {
      print("Error: $e");
      setState(() {
        _userId = "Request failed: $e";
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Face ID Scanner',
          style: TextStyle(color: Colors.green, fontSize: 24),
        ),
        centerTitle: true,
        backgroundColor: Colors.green.withOpacity(0.5),
      ),
      body: Stack(
        children: [
          if (_image == null)
            Center(
              child: ElevatedButton.icon(
                icon: Icon(
                  Icons.camera_alt_rounded,
                  size: 25,
                  color: Colors.black,
                ),
                label: Text('Capture Face', style: TextStyle(fontSize: 25)),
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  backgroundColor: Colors.tealAccent[400],
                  foregroundColor: Colors.black,
                  shape: StadiumBorder(),
                ),
                onPressed: _pickImage,
              ),
            )
          else
            Positioned(
              left: 0,
              right: 0,
              top: 100,
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    Hero(
                      tag: 'faceImage',
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.file(_image!, height: 250),
                      ),
                    ),
                    const SizedBox(height: 24),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 50.0,
                        vertical: 10,
                      ),
                      child: TextField(
                        controller: ipController,
                        decoration: InputDecoration(
                          labelText: "Enter IP Address",
                          border: OutlineInputBorder(),
                        ),
                        cursorColor: Colors.green,
                      ),
                    ),
                    ElevatedButton.icon(
                      icon: Icon(Icons.search, size: 20, color: Colors.black),
                      label: Text(
                        "Identify User",
                        style: TextStyle(fontSize: 16),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.black,
                        padding: EdgeInsets.symmetric(
                          horizontal: 24,
                          vertical: 12,
                        ),
                      ),
                      onPressed: _sendImageForIdentification,
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 50.0,
                        vertical: 10,
                      ),
                      child: TextField(
                        controller: userctrl,
                        decoration: InputDecoration(
                          labelText: "Enter User ID",
                          border: OutlineInputBorder(),
                        ),
                        cursorColor: Colors.green,
                      ),
                    ),
                    ElevatedButton.icon(
                      icon: Icon(Icons.fingerprint),
                      label: Text("Enroll User"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.teal[200],
                        foregroundColor: Colors.black,
                        padding: EdgeInsets.symmetric(
                          horizontal: 24,
                          vertical: 12,
                        ),
                      ),
                      onPressed: _sendImageForEnrollment,
                    ),
                    ElevatedButton.icon(
                      icon: Icon(Icons.redo),
                      label: Text("Take Another photo"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.teal[200],
                        foregroundColor: Colors.black,
                        padding: EdgeInsets.symmetric(
                          horizontal: 24,
                          vertical: 12,
                        ),
                      ),
                      onPressed: () {
                        setState(() {
                          _image = null;
                          _userId = "";
                        });
                      },
                    ),
                    if (_userId != null)
                      GlassContainer(
                        width: double.infinity,
                        height: 1000,
                        borderRadius: BorderRadius.circular(16),
                        blur: 20,
                        border: Border.all(color: Colors.tealAccent, width: 2),
                        gradient: LinearGradient(
                          colors: [
                            Colors.white.withOpacity(0.2),
                            Colors.white.withOpacity(0.05),
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        child: Text(
                          _userId!,
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 20, color: Colors.white),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          if (_loading)
            Container(
              color: Colors.black.withOpacity(0.5),
              child: Center(
                child: SpinKitFadingCube(color: Colors.green, size: 60),
              ),
            ),
        ],
      ),
    );
  }
}
