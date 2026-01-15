using System;
using System.Linq;
using System.Collections.Generic;
using System.Windows.Forms;              
using Microsoft.ML.OnnxRuntime;          
using Microsoft.ML.OnnxRuntime.Tensors;  
using OpenCvSharp;                       
using OpenCvSharp.Extensions;  
public class MorphDetectionForm : Form
{ Button btnSelect, btnDetect;
    PictureBox pictureBox;
    Label lblResult;
    string imagePath = "";
    string modelPath = "morph_autoencoder.onnx";
    public MorphDetectionForm()
    {   Text = "Morph Detection System";
        Width = 600;
        Height = 500;
        btnSelect = new Button { Text = "Select Image", Top = 20, Left = 20 };
        btnDetect = new Button { Text = "Detect Morph", Top = 20, Left = 140 };
        pictureBox = new PictureBox
        {     Top = 60,
            Left = 20,
            Width = 300,
            Height = 300,
            BorderStyle = BorderStyle.FixedSingle,
            SizeMode = PictureBoxSizeMode.StretchImage
        };lblResult = new Label
        {Top = 380,
            Left = 20,
            Width = 500,
            Font = new System.Drawing.Font("Arial", 14)
        };
        btnSelect.Click += SelectImage;
        btnDetect.Click += DetectMorph;
        Controls.Add(btnSelect);
        Controls.Add(btnDetect);
        Controls.Add(pictureBox);
        Controls.Add(lblResult);
    }
    void SelectImage(object sender, EventArgs e)
    {
        OpenFileDialog ofd = new OpenFileDialog();
        if (ofd.ShowDialog() == DialogResult.OK)
        {imagePath = ofd.FileName;
            pictureBox.Image = System.Drawing.Bitmap.FromFile(imagePath);
        }
    }
    void DetectMorph(object sender, EventArgs e)
    {if (string.IsNullOrEmpty(imagePath))
        {MessageBox.Show("Please select an image");
            return;
        }
        double error = RunModel(imagePath);
        lblResult.Text = error > 0.01
            ? "RESULT: MORPHED IMAGE"
            : "RESULT: GENUINE IMAGE";
    }
    double RunModel(string imagePath)
    {
        var session = new InferenceSession(modelPath);
        Mat img = Cv2.ImRead(imagePath);
        Cv2.Resize(img, img, new Size(224, 224));
        img.ConvertTo(img, MatType.CV_32FC3, 1.0 / 255);
        float[] inputData = new float[3 * 224 * 224];
        int idx = 0;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 224; y++)
                for (int x = 0; x < 224; x++)
                    inputData[idx++] = img.At<Vec3f>(y, x)[c];
        var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, 224, 224 });
        var inputs = new List<NamedOnnxValue>
        {    NamedOnnxValue.CreateFromTensor("input", tensor)
        };
        using var results = session.Run(inputs);
        float[] outputData = results.First().AsTensor<float>().ToArray();
        double error = 0;
        for (int i = 0; i < inputData.Length; i++)
            error += Math.Pow(inputData[i] - outputData[i], 2);
        return error / inputData.Length;
    }
    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Application.Run(new MorphDetectionForm());
    }
}
