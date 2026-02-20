package com.mgpu

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var promptInput: EditText
    private lateinit var outputText: TextView
    private lateinit var generateButton: Button
    private lateinit var loadingIndicator: ProgressBar
    private lateinit var latencyText: TextView

    companion object {
        init {
            System.loadLibrary("mgpu")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        promptInput = findViewById(R.id.prompt_input)
        outputText = findViewById(R.id.output_text)
        generateButton = findViewById(R.id.generate_button)
        loadingIndicator = findViewById(R.id.loading_indicator)
        latencyText = findViewById(R.id.latency_text)

        // Display device info
        val deviceInfo = getDeviceInfo()
        outputText.text = "Device:\n$deviceInfo\n\n"

        // Check for model in assets
        val modelFile = File(filesDir, "model.gguf")
        if (!modelFile.exists()) {
            copyAssetToFile("model.gguf", modelFile)
        }

        // Load model
        generateButton.isEnabled = false
        loadingIndicator.visibility = View.VISIBLE

        Thread {
            val kernelDir = filesDir.absolutePath + "/kernels"
            val success = loadModel(modelFile.absolutePath, kernelDir)

            runOnUiThread {
                loadingIndicator.visibility = View.GONE
                if (success) {
                    generateButton.isEnabled = true
                    outputText.append("Model loaded successfully!\n")
                } else {
                    outputText.append("Failed to load model!\n")
                }
            }
        }.start()

        // Generate button click
        generateButton.setOnClickListener {
            val prompt = promptInput.text.toString()
            if (prompt.isEmpty()) {
                Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            generateButton.isEnabled = false
            loadingIndicator.visibility = View.VISIBLE
            outputText.text = "Generating...\n"

            Thread {
                val result = generateText(prompt, 128)

                runOnUiThread {
                    loadingIndicator.visibility = View.GONE
                    generateButton.isEnabled = true
                    outputText.text = "Prompt: $prompt\n\nResult:\n$result\n"
                }
            }.start()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        unloadModel()
    }

    private fun copyAssetToFile(assetName: String, destFile: File) {
        try {
            assets.open(assetName).use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
            // Also copy kernel files
            val kernelDir = File(filesDir, "kernels")
            kernelDir.mkdirs()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // Native library interface
    external fun loadModel(modelPath: String, kernelDir: String): Boolean
    external fun generateText(prompt: String, maxTokens: Int): String
    external fun getDeviceInfo(): String
    external fun unloadModel()
}
