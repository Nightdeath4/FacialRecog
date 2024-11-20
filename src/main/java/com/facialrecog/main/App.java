package com.facialrecog.main;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class App {
    private static CascadeClassifier faceDetector;
    private static int currentFilter = 0;
    private static Mat frame;

    static {
        try {
            nu.pattern.OpenCV.loadLocally();
            frame = new Mat();
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV library: " + e.getMessage());
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        JFrame window = new JFrame("Facial Recognition");
        JLabel screen = new JLabel();
        window.add(screen);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setSize(640, 480);

        faceDetector = new CascadeClassifier();
        String cascadePath = "src/main/resources/haarcascade_frontalface_alt.xml";
        faceDetector.load(cascadePath);

        VideoCapture capture = new VideoCapture(0);
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);

        if (!capture.isOpened()) {
            System.out.println("Error: Camera not accessible");
            return;
        }

        window.setVisible(true);

        window.addKeyListener(new KeyListener() {
            @Override
            public void keyPressed(KeyEvent e) {
                String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
                
                if (e.getKeyCode() == KeyEvent.VK_SPACE) {
                    String filename = "capture_" + timestamp + ".jpg";
                    Imgcodecs.imwrite(filename, frame);
                    System.out.println("Full image saved as: " + filename);
                }
                else if (e.getKeyCode() == KeyEvent.VK_F) {
                    MatOfRect faceDetections = new MatOfRect();
                    faceDetector.detectMultiScale(frame, faceDetections);
                    Rect[] faces = faceDetections.toArray();
                    
                    for (int i = 0; i < faces.length; i++) {
                        Mat face = new Mat(frame, faces[i]);
                        String filename = "face_" + timestamp + "_" + (i + 1) + ".jpg";
                        Imgcodecs.imwrite(filename, face);
                        System.out.println("Face " + (i + 1) + " saved as: " + filename);
                    }
                }
                else if (e.getKeyCode() >= KeyEvent.VK_1 && e.getKeyCode() <= KeyEvent.VK_4) {
                    currentFilter = e.getKeyCode() - KeyEvent.VK_1;
                    System.out.println("Filter changed to: " + currentFilter);
                }
            }

            @Override
            public void keyTyped(KeyEvent e) {}

            @Override
            public void keyReleased(KeyEvent e) {}
        });

        window.setFocusable(true);
        window.requestFocus();

        while (true) {
            capture.read(frame);
            if (frame.empty()) break;

            switch (currentFilter) {
                case 1: // Grayscale
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2BGR);
                    break;
                case 2: // Blur
                    Imgproc.GaussianBlur(frame, frame, new org.opencv.core.Size(15, 15), 0);
                    break;
                case 3: // Edge Detection
                    Mat edges = new Mat();
                    Imgproc.Canny(frame, edges, 100, 200);
                    frame = edges;
                    break;
                default: // No filter
                    break;
            }

            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(frame, faceDetections);

            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(
                    frame,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0),
                    2
                );
            }

            BufferedImage image = matToBufferedImage(frame);
            screen.setIcon(new ImageIcon(image));
            window.repaint();
        }

        capture.release();
        window.dispose();
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        mat.get(0, 0, pixels);
        return image;
    }
}
