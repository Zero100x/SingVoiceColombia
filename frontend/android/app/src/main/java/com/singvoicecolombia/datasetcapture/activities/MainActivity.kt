package com.singvoicecolombia.datasetcapture.activities

import android.Manifest
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.singvoicecolombia.datasetcapture.R
import com.singvoicecolombia.datasetcapture.services.AlphabetRecognitionService
import com.singvoicecolombia.datasetcapture.services.DynamicRecognitionService
import com.singvoicecolombia.datasetcapture.utils.PermissionHelper
import com.singvoicecolombia.datasetcapture.utils.VoiceOutputManager
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var textResultCardTitle: TextView
    private lateinit var textDetectedLetter: TextView
    private lateinit var textConfidence: TextView
    private lateinit var textStatus: TextView
    private lateinit var buttonToggleCamera: MaterialButton
    private lateinit var buttonSwitchCamera: MaterialButton
    private lateinit var buttonSpeak: MaterialButton
    private lateinit var buttonRecognitionMode: MaterialButton
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var recognitionService: AlphabetRecognitionService
    private lateinit var dynamicRecognitionService: DynamicRecognitionService
    private lateinit var voiceOutputManager: VoiceOutputManager

    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraEnabled = false
    private var currentLensFacing = CameraSelector.LENS_FACING_BACK
    private var recognitionMode = RecognitionMode.STATIC
    private var lastLetter = ""
    private var lastAnalyzedAt = 0L

    private val requestCameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            startCamera()
        } else {
            showPermissionDeniedDialog()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewViewMain)
        textResultCardTitle = findViewById(R.id.textResultCardTitle)
        textDetectedLetter = findViewById(R.id.textDetectedLetter)
        textConfidence = findViewById(R.id.textConfidence)
        textStatus = findViewById(R.id.textRecognitionStatus)
        buttonToggleCamera = findViewById(R.id.buttonToggleCamera)
        buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera)
        buttonSpeak = findViewById(R.id.buttonSpeakResult)
        buttonRecognitionMode = findViewById(R.id.buttonRecognitionMode)

        cameraExecutor = Executors.newSingleThreadExecutor()
        recognitionService = AlphabetRecognitionService(this)
        dynamicRecognitionService = DynamicRecognitionService(this)
        voiceOutputManager = VoiceOutputManager(this)

        renderInitialState()

        buttonToggleCamera.setOnClickListener {
            if (cameraEnabled) {
                stopCamera()
            } else {
                checkCameraPermissionAndStart()
            }
        }

        buttonSwitchCamera.setOnClickListener {
            switchCamera()
        }

        buttonRecognitionMode.setOnClickListener {
            switchRecognitionMode()
        }

        buttonSpeak.setOnClickListener {
            if (lastLetter.isBlank()) {
                Toast.makeText(this, R.string.voice_no_result_message, Toast.LENGTH_SHORT).show()
            } else {
                val textToSpeak = if (recognitionMode == RecognitionMode.STATIC) {
                    getString(R.string.voice_letter_result, lastLetter)
                } else {
                    getString(R.string.voice_sign_result, lastLetter)
                }
                voiceOutputManager.speak(textToSpeak)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopCamera()
        recognitionService.close()
        dynamicRecognitionService.close()
        voiceOutputManager.shutdown()
        cameraExecutor.shutdown()
    }

    private fun renderInitialState() {
        textResultCardTitle.text = if (recognitionMode == RecognitionMode.STATIC) {
            getString(R.string.result_card_title_static)
        } else {
            getString(R.string.result_card_title_dynamic)
        }
        textDetectedLetter.text = getString(R.string.detected_letter_empty)
        textConfidence.text = getString(R.string.confidence_empty)
        textStatus.text = if (isCurrentModelReady()) {
            getString(R.string.status_waiting)
        } else {
            getCurrentModelPendingText()
        }
        buttonToggleCamera.text = if (cameraEnabled) {
            getString(R.string.camera_disable_button)
        } else {
            getString(R.string.camera_enable_button)
        }
        buttonRecognitionMode.text = if (recognitionMode == RecognitionMode.STATIC) {
            getString(R.string.recognition_mode_static)
        } else {
            getString(R.string.recognition_mode_dynamic)
        }
    }

    private fun isCurrentModelReady(): Boolean {
        return if (recognitionMode == RecognitionMode.STATIC) {
            recognitionService.isModelReady()
        } else {
            dynamicRecognitionService.isModelReady()
        }
    }

    private fun getCurrentModelPendingText(): String {
        return if (recognitionMode == RecognitionMode.STATIC) {
            getString(R.string.status_model_pending)
        } else {
            getString(R.string.status_dynamic_model_pending)
        }
    }

    private fun checkCameraPermissionAndStart() {
        if (PermissionHelper.hasCameraPermission(this)) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            cameraProvider = provider

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        val now = System.currentTimeMillis()
                        if (now - lastAnalyzedAt < currentAnalysisIntervalMs()) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        lastAnalyzedAt = now
                        analyzeFrame(imageProxy)
                    }
                }

            try {
                provider.unbindAll()
                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(currentLensFacing)
                    .build()

                if (!provider.hasCamera(cameraSelector)) {
                    Toast.makeText(this, R.string.camera_unavailable_message, Toast.LENGTH_SHORT).show()
                    currentLensFacing = oppositeLensFacing()
                    return@addListener
                }

                provider.bindToLifecycle(this, cameraSelector, preview, analyzer)
                cameraEnabled = true
                buttonToggleCamera.text = getString(R.string.camera_disable_button)
                textStatus.text = if (isCurrentModelReady()) {
                    getString(R.string.status_recognizing)
                } else {
                    getCurrentModelPendingText()
                }
            } catch (_: Exception) {
                Toast.makeText(this, R.string.camera_start_error, Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraEnabled = false
        if (::buttonToggleCamera.isInitialized) {
            buttonToggleCamera.text = getString(R.string.camera_enable_button)
        }
        if (::textStatus.isInitialized) {
            textStatus.text = getString(R.string.status_waiting)
        }
    }

    private fun switchRecognitionMode() {
        recognitionMode = if (recognitionMode == RecognitionMode.STATIC) {
            RecognitionMode.DYNAMIC
        } else {
            RecognitionMode.STATIC
        }
        lastLetter = ""
        lastAnalyzedAt = 0L
        dynamicRecognitionService.reset()
        renderInitialState()

        if (cameraEnabled) {
            textStatus.text = if (isCurrentModelReady()) {
                getString(R.string.status_recognizing)
            } else {
                getCurrentModelPendingText()
            }
        }
    }

    private fun switchCamera() {
        currentLensFacing = oppositeLensFacing()
        lastAnalyzedAt = 0L

        if (cameraEnabled) {
            startCamera()
        } else {
            Toast.makeText(this, R.string.camera_switch_button, Toast.LENGTH_SHORT).show()
        }
    }

    private fun oppositeLensFacing(): Int {
        return if (currentLensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
    }

    private fun currentAnalysisIntervalMs(): Long {
        return if (recognitionMode == RecognitionMode.STATIC) {
            STATIC_ANALYSIS_INTERVAL_MS
        } else {
            DYNAMIC_ANALYSIS_INTERVAL_MS
        }
    }

    private fun analyzeFrame(imageProxy: androidx.camera.core.ImageProxy) {
        if (recognitionMode == RecognitionMode.STATIC) {
            analyzeStaticFrame(imageProxy)
        } else {
            analyzeDynamicFrame(imageProxy)
        }
    }

    private fun analyzeStaticFrame(imageProxy: androidx.camera.core.ImageProxy) {
        val result = recognitionService.classify(imageProxy)

        runOnUiThread {
            if (result == null) {
                textStatus.text = getCurrentModelPendingText()
                return@runOnUiThread
            }

            if (!result.signPresent) {
                lastLetter = ""
                textDetectedLetter.text = getString(R.string.detected_letter_empty)
                textConfidence.text = getString(R.string.confidence_empty)
                textStatus.text = getString(R.string.status_no_hand_detected)
                return@runOnUiThread
            }

            textConfidence.text = getString(R.string.confidence_value, result.confidence * 100f)

            if (
                isNoSignLabel(result.letter) ||
                result.confidence < MIN_RECOGNITION_CONFIDENCE ||
                result.margin < MIN_RECOGNITION_MARGIN
            ) {
                lastLetter = ""
                textDetectedLetter.text = getString(R.string.detected_letter_empty)
                textStatus.text = getString(R.string.status_unrecognized)
            } else {
                lastLetter = result.letter
                textDetectedLetter.text = result.letter
                textStatus.text = getString(R.string.status_detected)
            }
        }
    }

    private fun isNoSignLabel(label: String): Boolean {
        return when (label.trim().uppercase()) {
            "NO_SENA", "SIN_SENA", "FONDO", "BACKGROUND", "NINGUNA" -> true
            else -> false
        }
    }

    private fun analyzeDynamicFrame(imageProxy: androidx.camera.core.ImageProxy) {
        val result = dynamicRecognitionService.classify(imageProxy)

        runOnUiThread {
            if (result == null) {
                textStatus.text = getCurrentModelPendingText()
                return@runOnUiThread
            }

            if (!result.ready) {
                lastLetter = ""
                textDetectedLetter.text = getString(R.string.detected_letter_empty)
                textConfidence.text = getString(R.string.confidence_empty)
                textStatus.text = getString(
                    R.string.status_dynamic_collecting,
                    result.collectedFrames,
                    result.requiredFrames
                )
                return@runOnUiThread
            }

            textConfidence.text = getString(R.string.confidence_value, result.confidence * 100f)

            if (result.confidence < MIN_DYNAMIC_RECOGNITION_CONFIDENCE) {
                lastLetter = ""
                textDetectedLetter.text = getString(R.string.detected_letter_empty)
                textStatus.text = getString(R.string.status_unrecognized)
            } else {
                lastLetter = result.label
                textDetectedLetter.text = result.label
                textStatus.text = getString(R.string.status_detected)
            }
        }
    }

    private fun showPermissionDeniedDialog() {
        AlertDialog.Builder(this)
            .setTitle(R.string.camera_permission_required_title)
            .setMessage(R.string.camera_permission_required_message)
            .setPositiveButton(R.string.accept_button_label, null)
            .show()
    }

    companion object {
        private const val STATIC_ANALYSIS_INTERVAL_MS = 450L
        private const val DYNAMIC_ANALYSIS_INTERVAL_MS = 120L
        private const val MIN_RECOGNITION_CONFIDENCE = 0.50f
        private const val MIN_RECOGNITION_MARGIN = 0.06f
        private const val MIN_DYNAMIC_RECOGNITION_CONFIDENCE = 0.65f
    }

    private enum class RecognitionMode {
        STATIC,
        DYNAMIC
    }
}
