package com.singvoicecolombia.datasetcapture.activities

import android.Manifest
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
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
import java.util.ArrayDeque
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var sectionHome: View
    private lateinit var sectionTranslate: View
    private lateinit var sectionLearn: View
    private lateinit var sectionHistory: View
    private lateinit var sectionProfile: View
    private lateinit var bottomNavigation: View
    private lateinit var cameraPlaceholder: View
    private lateinit var navHome: LinearLayout
    private lateinit var navLearn: LinearLayout
    private lateinit var navHistory: LinearLayout
    private lateinit var navProfile: LinearLayout
    private lateinit var iconNavHome: ImageView
    private lateinit var iconNavLearn: ImageView
    private lateinit var iconNavHistory: ImageView
    private lateinit var iconNavProfile: ImageView
    private lateinit var textNavHome: TextView
    private lateinit var textNavLearn: TextView
    private lateinit var textNavHistory: TextView
    private lateinit var textNavProfile: TextView
    private lateinit var previewView: PreviewView
    private lateinit var textTopStatus: TextView
    private lateinit var textCameraHint: TextView
    private lateinit var reticleGuide: View
    private lateinit var textGestureMode: TextView
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
    private var currentLensFacing = CameraSelector.LENS_FACING_FRONT
    private var recognitionMode = RecognitionMode.STATIC
    private var activeSection = AppSection.HOME
    private var lastLetter = ""
    private var lastDynamicTranslation = ""
    private var lastDynamicConfidence = 0f
    private var lastDynamicDetectedAt = 0L
    private var lastAnalyzedAt = 0L
    private var autoSpeakEnabled = false
    private var lastSpokenText = ""
    private var lastSpokenAt = 0L
    private val staticPredictionVotes = ArrayDeque<StaticPredictionVote>()
    private val dynamicPredictionVotes = ArrayDeque<DynamicPredictionVote>()

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

        sectionHome = findViewById(R.id.sectionHome)
        sectionTranslate = findViewById(R.id.sectionTranslate)
        sectionLearn = findViewById(R.id.sectionLearn)
        sectionHistory = findViewById(R.id.sectionHistory)
        sectionProfile = findViewById(R.id.sectionProfile)
        bottomNavigation = findViewById(R.id.bottomNavigation)
        cameraPlaceholder = findViewById(R.id.cameraPlaceholder)
        navHome = findViewById(R.id.navHome)
        navLearn = findViewById(R.id.navLearn)
        navHistory = findViewById(R.id.navHistory)
        navProfile = findViewById(R.id.navProfile)
        iconNavHome = findViewById(R.id.iconNavHome)
        iconNavLearn = findViewById(R.id.iconNavLearn)
        iconNavHistory = findViewById(R.id.iconNavHistory)
        iconNavProfile = findViewById(R.id.iconNavProfile)
        textNavHome = findViewById(R.id.textNavHome)
        textNavLearn = findViewById(R.id.textNavLearn)
        textNavHistory = findViewById(R.id.textNavHistory)
        textNavProfile = findViewById(R.id.textNavProfile)
        previewView = findViewById(R.id.previewViewMain)
        textTopStatus = findViewById(R.id.textTranslatorFrom)
        textCameraHint = findViewById(R.id.textCameraHint)
        reticleGuide = findViewById(R.id.reticleGuide)
        textGestureMode = findViewById(R.id.textGestureMode)
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

        setupNavigation()
        renderInitialState()
        showSection(AppSection.HOME, animate = false)
        showTermsIfNeeded()

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
            toggleAutoSpeak()
        }
    }

    private fun setupNavigation() {
        findViewById<View>(R.id.cardTranslate).setOnClickListener {
            animatePress(it)
            showSection(AppSection.TRANSLATE)
        }
        findViewById<View>(R.id.cardLearn).setOnClickListener {
            animatePress(it)
            showSection(AppSection.LEARN)
        }
        findViewById<View>(R.id.cardHistory).setOnClickListener {
            animatePress(it)
            showSection(AppSection.HISTORY)
        }
        findViewById<View>(R.id.buttonShowTerms).setOnClickListener {
            showTermsDialog(storeAcceptance = false)
        }
        findViewById<View>(R.id.buttonBackToHome).setOnClickListener {
            showSection(AppSection.HOME)
        }

        navHome.setOnClickListener { showSection(AppSection.HOME) }
        navLearn.setOnClickListener { showSection(AppSection.LEARN) }
        navHistory.setOnClickListener { showSection(AppSection.HISTORY) }
        navProfile.setOnClickListener { showSection(AppSection.PROFILE) }
    }

    private fun showSection(section: AppSection, animate: Boolean = true) {
        if (section != AppSection.TRANSLATE && cameraEnabled) {
            stopCamera()
        }

        activeSection = section
        val sections = mapOf(
            AppSection.HOME to sectionHome,
            AppSection.TRANSLATE to sectionTranslate,
            AppSection.LEARN to sectionLearn,
            AppSection.HISTORY to sectionHistory,
            AppSection.PROFILE to sectionProfile
        )

        sections.values.forEach { it.visibility = View.GONE }
        val selectedView = sections.getValue(section)
        selectedView.visibility = View.VISIBLE
        bottomNavigation.visibility = if (section == AppSection.TRANSLATE) View.GONE else View.VISIBLE
        if (animate) animateSectionIn(selectedView)
        updateBottomNavigation(section)
        renderInitialState()
    }

    private fun animateSectionIn(view: View) {
        view.alpha = 0f
        view.translationY = 22f
        view.animate()
            .alpha(1f)
            .translationY(0f)
            .setDuration(220L)
            .start()
    }

    private fun animatePress(view: View) {
        view.animate()
            .scaleX(0.98f)
            .scaleY(0.98f)
            .setDuration(80L)
            .withEndAction {
                view.animate()
                    .scaleX(1f)
                    .scaleY(1f)
                    .setDuration(120L)
                    .start()
            }
            .start()
    }

    private fun updateBottomNavigation(section: AppSection) {
        val selectedColor = ContextCompat.getColor(this, R.color.blue_action)
        val normalColor = ContextCompat.getColor(this, R.color.text_muted)

        setNavItemState(navHome, iconNavHome, textNavHome, section == AppSection.HOME, selectedColor, normalColor)
        setNavItemState(navLearn, iconNavLearn, textNavLearn, section == AppSection.LEARN, selectedColor, normalColor)
        setNavItemState(navHistory, iconNavHistory, textNavHistory, section == AppSection.HISTORY, selectedColor, normalColor)
        setNavItemState(navProfile, iconNavProfile, textNavProfile, section == AppSection.PROFILE, selectedColor, normalColor)
    }

    private fun setNavItemState(
        container: LinearLayout,
        icon: ImageView,
        label: TextView,
        selected: Boolean,
        selectedColor: Int,
        normalColor: Int
    ) {
        val color = if (selected) selectedColor else normalColor
        container.background = if (selected) {
            ContextCompat.getDrawable(this, R.drawable.bg_nav_selected)
        } else {
            null
        }
        icon.setColorFilter(color)
        label.setTextColor(color)
        label.setTypeface(null, if (selected) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
    }

    private fun showTermsIfNeeded() {
        val accepted = getSharedPreferences(PREFERENCES_NAME, MODE_PRIVATE)
            .getBoolean(PREF_TERMS_ACCEPTED, false)
        if (!accepted) {
            showTermsDialog(storeAcceptance = true)
        }
    }

    private fun showTermsDialog(storeAcceptance: Boolean) {
        AlertDialog.Builder(this)
            .setTitle(R.string.terms_screen_title)
            .setMessage(R.string.terms_screen_body)
            .setCancelable(!storeAcceptance)
            .setPositiveButton(R.string.terms_accept_button) { _, _ ->
                if (storeAcceptance) {
                    getSharedPreferences(PREFERENCES_NAME, MODE_PRIVATE)
                        .edit()
                        .putBoolean(PREF_TERMS_ACCEPTED, true)
                        .apply()
                }
            }
            .setNegativeButton(if (storeAcceptance) R.string.terms_exit_button else R.string.back_button_label) { _, _ ->
                if (storeAcceptance) finish()
            }
            .show()
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
        textDetectedLetter.text = getString(R.string.translation_placeholder)
        textConfidence.text = getString(R.string.confidence_empty)
        textStatus.text = if (isCurrentModelReady()) {
            getString(R.string.status_waiting)
        } else {
            getCurrentModelPendingText()
        }
        val cameraActionLabel = if (cameraEnabled) {
            getString(R.string.stop_translation_button)
        } else {
            getString(R.string.start_translation_button)
        }
        buttonToggleCamera.text = ""
        buttonToggleCamera.contentDescription = cameraActionLabel
        buttonToggleCamera.setIconResource(
            if (cameraEnabled) android.R.drawable.ic_media_pause else android.R.drawable.ic_media_play
        )

        val recognitionModeLabel = if (recognitionMode == RecognitionMode.STATIC) {
            getString(R.string.recognition_mode_static)
        } else {
            getString(R.string.recognition_mode_dynamic)
        }
        buttonRecognitionMode.text = ""
        buttonRecognitionMode.contentDescription = recognitionModeLabel
        buttonSwitchCamera.text = ""
        buttonSwitchCamera.contentDescription = getString(R.string.camera_switch_button)
        updateAutoSpeakButton()
        updateRecognitionVisualState(DetectionVisualState.WAITING)
    }

    private fun updateRecognitionVisualState(state: DetectionVisualState) {
        if (!::textTopStatus.isInitialized) return

        textTopStatus.text = when (state) {
            DetectionVisualState.WAITING -> getString(R.string.translation_status_waiting)
            DetectionVisualState.RECOGNIZING -> getString(R.string.translation_status_recognizing)
            DetectionVisualState.DETECTED -> getString(R.string.translation_status_detected)
            DetectionVisualState.RETRY -> getString(R.string.translation_status_retry)
        }

        textCameraHint.visibility = if (state == DetectionVisualState.WAITING) {
            View.VISIBLE
        } else {
            View.GONE
        }

        reticleGuide.background = ContextCompat.getDrawable(
            this,
            when (state) {
                DetectionVisualState.WAITING -> R.drawable.bg_translate_reticle_idle
                DetectionVisualState.RECOGNIZING -> R.drawable.bg_translate_reticle_active
                DetectionVisualState.DETECTED -> R.drawable.bg_translate_reticle_detected
                DetectionVisualState.RETRY -> R.drawable.bg_translate_reticle_retry
            }
        )

        textGestureMode.text = when {
            !cameraEnabled -> getString(R.string.gesture_mode_waiting)
            recognitionMode == RecognitionMode.STATIC -> getString(R.string.gesture_mode_static)
            else -> getString(R.string.gesture_mode_dynamic)
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
            if (!recognitionService.isAlphabetModelReady()) {
                getString(R.string.status_model_pending)
            } else {
                getString(R.string.status_hand_detector_pending)
            }
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
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
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
                cameraPlaceholder.visibility = View.GONE
                buttonToggleCamera.text = ""
                buttonToggleCamera.contentDescription = getString(R.string.stop_translation_button)
                buttonToggleCamera.setIconResource(android.R.drawable.ic_media_pause)
                textStatus.text = if (isCurrentModelReady()) {
                    getString(R.string.status_recognizing)
                } else {
                    getCurrentModelPendingText()
                }
                updateRecognitionVisualState(
                    if (isCurrentModelReady()) DetectionVisualState.RECOGNIZING else DetectionVisualState.RETRY
                )
            } catch (_: Exception) {
                Toast.makeText(this, R.string.camera_start_error, Toast.LENGTH_LONG).show()
                updateRecognitionVisualState(DetectionVisualState.RETRY)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraEnabled = false
        clearStaticPredictionVotes()
        clearDynamicPredictionVotes()
        if (::buttonToggleCamera.isInitialized) {
            buttonToggleCamera.text = ""
            buttonToggleCamera.contentDescription = getString(R.string.start_translation_button)
            buttonToggleCamera.setIconResource(android.R.drawable.ic_media_play)
        }
        if (::cameraPlaceholder.isInitialized) {
            cameraPlaceholder.visibility = View.VISIBLE
        }
        if (::textStatus.isInitialized) {
            textStatus.text = getString(R.string.status_waiting)
        }
        if (::textTopStatus.isInitialized) {
            updateRecognitionVisualState(DetectionVisualState.WAITING)
        }
    }

    private fun switchRecognitionMode() {
        recognitionMode = if (recognitionMode == RecognitionMode.STATIC) {
            RecognitionMode.DYNAMIC
        } else {
            RecognitionMode.STATIC
        }
        lastLetter = ""
        clearDynamicTranslation()
        lastAnalyzedAt = 0L
        clearStaticPredictionVotes()
        clearDynamicPredictionVotes()
        dynamicRecognitionService.reset()
        renderInitialState()

        if (cameraEnabled) {
            textStatus.text = if (isCurrentModelReady()) {
                getString(R.string.status_recognizing)
            } else {
                getCurrentModelPendingText()
            }
            updateRecognitionVisualState(
                if (isCurrentModelReady()) DetectionVisualState.RECOGNIZING else DetectionVisualState.RETRY
            )
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
        val result = recognitionService.classify(
            imageProxy,
            mirrorHorizontally = currentLensFacing == CameraSelector.LENS_FACING_FRONT
        )

        runOnUiThread {
            if (result == null) {
                clearStaticPredictionVotes()
                textStatus.text = getCurrentModelPendingText()
                updateRecognitionVisualState(DetectionVisualState.RETRY)
                return@runOnUiThread
            }

            if (!result.signPresent) {
                clearStaticPredictionVotes()
                lastLetter = ""
                textDetectedLetter.text = getString(R.string.translation_placeholder)
                textConfidence.text = getString(R.string.confidence_empty)
                textStatus.text = getString(R.string.status_no_hand_detected)
                updateRecognitionVisualState(DetectionVisualState.WAITING)
                return@runOnUiThread
            }

            textConfidence.text = getString(R.string.confidence_value, result.confidence * 100f)

            if (
                isNoSignLabel(result.letter) ||
                result.confidence < MIN_RECOGNITION_CONFIDENCE ||
                result.margin < MIN_RECOGNITION_MARGIN
            ) {
                clearStaticPredictionVotes()
                lastLetter = ""
                textDetectedLetter.text = getString(R.string.translation_placeholder)
                textStatus.text = getString(R.string.status_unrecognized)
                updateRecognitionVisualState(DetectionVisualState.RETRY)
            } else {
                rememberStaticPrediction(result.letter, result.confidence)
                val stablePrediction = stableStaticPrediction()

                if (isHighConfidenceStaticPrediction(result)) {
                    val displayLabel = displayStaticLabel(result.letter)
                    lastLetter = displayLabel
                    textDetectedLetter.text = displayLabel
                    textConfidence.text = getString(
                        R.string.confidence_value,
                        result.confidence * 100f
                    )
                    textStatus.text = getString(R.string.status_letter_detected)
                    updateRecognitionVisualState(DetectionVisualState.DETECTED)
                    speakRecognizedText(displayLabel)
                } else if (stablePrediction == null) {
                    lastLetter = ""
                    textDetectedLetter.text = getString(R.string.translation_placeholder)
                    textStatus.text = getString(R.string.status_recognizing)
                    updateRecognitionVisualState(DetectionVisualState.RECOGNIZING)
                } else {
                    val displayLabel = displayStaticLabel(stablePrediction.label)
                    lastLetter = displayLabel
                    textDetectedLetter.text = displayLabel
                    textConfidence.text = getString(
                        R.string.confidence_value,
                        stablePrediction.confidence * 100f
                    )
                    textStatus.text = getString(R.string.status_letter_detected)
                    updateRecognitionVisualState(DetectionVisualState.DETECTED)
                    speakRecognizedText(displayLabel)
                }
            }
        }
    }

    private fun isHighConfidenceStaticPrediction(result: com.singvoicecolombia.datasetcapture.services.AlphabetRecognitionResult): Boolean {
        return result.confidence >= HIGH_CONFIDENCE_STATIC_THRESHOLD &&
            result.margin >= HIGH_CONFIDENCE_STATIC_MARGIN
    }

    private fun rememberStaticPrediction(label: String, confidence: Float) {
        staticPredictionVotes.addLast(StaticPredictionVote(label, confidence))
        while (staticPredictionVotes.size > STATIC_STABILITY_WINDOW) {
            staticPredictionVotes.removeFirst()
        }
    }

    private fun clearStaticPredictionVotes() {
        staticPredictionVotes.clear()
    }

    private fun stableStaticPrediction(): StaticPredictionVote? {
        if (staticPredictionVotes.size < MIN_STATIC_STABLE_VOTES) return null

        val votesByLabel = staticPredictionVotes.groupBy { it.label }
        val mostVoted = votesByLabel.maxByOrNull { it.value.size } ?: return null
        val votes = mostVoted.value
        val stableRatio = votes.size.toFloat() / staticPredictionVotes.size.toFloat()
        val averageConfidence = votes.map { it.confidence }.average().toFloat()

        if (
            votes.size < MIN_STATIC_STABLE_VOTES ||
            stableRatio < MIN_STATIC_STABLE_RATIO ||
            averageConfidence < MIN_RECOGNITION_CONFIDENCE
        ) {
            return null
        }

        return StaticPredictionVote(mostVoted.key, averageConfidence)
    }

    private fun displayStaticLabel(label: String): String {
        return if (label.trim().uppercase() == "NN") {
            "\u00D1"
        } else {
            label
        }
    }

    private fun isNoSignLabel(label: String): Boolean {
        return when (label.trim().uppercase()) {
            "NO_SENA", "SIN_SENA", "FONDO", "BACKGROUND", "NINGUNA" -> true
            "DESCONOCIDO", "UNKNOWN", "NO_ENTRENADA", "SENA_NO_ENTRENADA" -> true
            else -> false
        }
    }

    private fun analyzeDynamicFrame(imageProxy: androidx.camera.core.ImageProxy) {
        val result = dynamicRecognitionService.classify(
            imageProxy,
            mirrorHorizontally = currentLensFacing == CameraSelector.LENS_FACING_FRONT
        )

        runOnUiThread {
            if (result == null) {
                clearDynamicPredictionVotes()
                textStatus.text = getCurrentModelPendingText()
                updateRecognitionVisualState(DetectionVisualState.RETRY)
                return@runOnUiThread
            }

            if (!result.handPresent) {
                clearDynamicPredictionVotes()
                showHeldDynamicTranslationOrPlaceholder()
                textStatus.text = getString(R.string.status_no_hand_detected)
                updateRecognitionVisualState(DetectionVisualState.WAITING)
                return@runOnUiThread
            }

            if (!result.ready) {
                showHeldDynamicTranslationOrPlaceholder()
                textStatus.text = getString(
                    R.string.status_dynamic_collecting,
                    result.collectedFrames,
                    result.requiredFrames
                )
                updateRecognitionVisualState(DetectionVisualState.RECOGNIZING)
                return@runOnUiThread
            }

            if (!result.hasMotion) {
                clearDynamicPredictionVotes()
                showHeldDynamicTranslationOrPlaceholder()
                textStatus.text = getString(R.string.status_dynamic_no_motion)
                updateRecognitionVisualState(DetectionVisualState.RECOGNIZING)
                return@runOnUiThread
            }

            textConfidence.text = getString(R.string.confidence_value, result.confidence * 100f)

            if (isNoSignLabel(result.label)) {
                clearDynamicPredictionVotes()
                showHeldDynamicTranslationOrPlaceholder()
                textStatus.text = getString(R.string.status_dynamic_unknown_sign)
                updateRecognitionVisualState(DetectionVisualState.RETRY)
            } else if (
                result.confidence < MIN_DYNAMIC_RECOGNITION_CONFIDENCE ||
                result.margin < MIN_DYNAMIC_RECOGNITION_MARGIN
            ) {
                clearDynamicPredictionVotes()
                showHeldDynamicTranslationOrPlaceholder()
                textStatus.text = getString(R.string.status_unrecognized)
                updateRecognitionVisualState(DetectionVisualState.RETRY)
            } else {
                rememberDynamicPrediction(result.label, result.confidence)
                val stablePrediction = stableDynamicPrediction()

                if (stablePrediction == null) {
                    showHeldDynamicTranslationOrPlaceholder()
                    textStatus.text = getString(R.string.status_recognizing)
                    updateRecognitionVisualState(DetectionVisualState.RECOGNIZING)
                } else {
                    rememberDynamicTranslation(stablePrediction.label, stablePrediction.confidence)
                    lastLetter = lastDynamicTranslation
                    textDetectedLetter.text = lastDynamicTranslation
                    textConfidence.text = getString(
                        R.string.confidence_value,
                        stablePrediction.confidence * 100f
                    )
                    textStatus.text = getString(R.string.status_detected)
                    updateRecognitionVisualState(DetectionVisualState.DETECTED)
                    speakRecognizedText(lastDynamicTranslation)
                }
            }
        }
    }

    private fun rememberDynamicTranslation(label: String, confidence: Float) {
        lastDynamicTranslation = label
        lastDynamicConfidence = confidence
        lastDynamicDetectedAt = System.currentTimeMillis()
    }

    private fun rememberDynamicPrediction(label: String, confidence: Float) {
        dynamicPredictionVotes.addLast(DynamicPredictionVote(label, confidence))
        while (dynamicPredictionVotes.size > DYNAMIC_STABILITY_WINDOW) {
            dynamicPredictionVotes.removeFirst()
        }
    }

    private fun stableDynamicPrediction(): DynamicPredictionVote? {
        if (dynamicPredictionVotes.size < MIN_DYNAMIC_STABLE_VOTES) return null

        val votesByLabel = dynamicPredictionVotes.groupBy { it.label }
        val mostVoted = votesByLabel.maxByOrNull { it.value.size } ?: return null
        val votes = mostVoted.value
        val stableRatio = votes.size.toFloat() / dynamicPredictionVotes.size.toFloat()
        val averageConfidence = votes.map { it.confidence }.average().toFloat()

        if (
            votes.size < MIN_DYNAMIC_STABLE_VOTES ||
            stableRatio < MIN_DYNAMIC_STABLE_RATIO ||
            averageConfidence < MIN_DYNAMIC_RECOGNITION_CONFIDENCE
        ) {
            return null
        }

        return DynamicPredictionVote(mostVoted.key, averageConfidence)
    }

    private fun clearDynamicPredictionVotes() {
        dynamicPredictionVotes.clear()
    }

    private fun clearDynamicTranslation() {
        lastDynamicTranslation = ""
        lastDynamicConfidence = 0f
        lastDynamicDetectedAt = 0L
    }

    private fun showHeldDynamicTranslationOrPlaceholder() {
        if (shouldKeepDynamicTranslationVisible()) {
            lastLetter = lastDynamicTranslation
            textDetectedLetter.text = lastDynamicTranslation
            textConfidence.text = getString(
                R.string.confidence_value,
                lastDynamicConfidence * 100f
            )
        } else {
            lastLetter = ""
            textDetectedLetter.text = getString(R.string.translation_placeholder)
            textConfidence.text = getString(R.string.confidence_empty)
        }
    }

    private fun shouldKeepDynamicTranslationVisible(): Boolean {
        return lastDynamicTranslation.isNotBlank() &&
            System.currentTimeMillis() - lastDynamicDetectedAt <= DYNAMIC_RESULT_HOLD_MS
    }

    private fun toggleAutoSpeak() {
        autoSpeakEnabled = !autoSpeakEnabled
        lastSpokenText = ""
        lastSpokenAt = 0L
        if (!autoSpeakEnabled) {
            voiceOutputManager.stop()
        }
        updateAutoSpeakButton()

        val message = if (autoSpeakEnabled) {
            getString(R.string.audio_auto_on)
        } else {
            getString(R.string.audio_auto_off)
        }
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun updateAutoSpeakButton() {
        if (!::buttonSpeak.isInitialized) return

        val label = if (autoSpeakEnabled) {
            getString(R.string.audio_auto_on)
        } else {
            getString(R.string.audio_auto_off)
        }
        buttonSpeak.text = ""
        buttonSpeak.contentDescription = label
        buttonSpeak.setIconResource(
            if (autoSpeakEnabled) {
                android.R.drawable.ic_lock_silent_mode_off
            } else {
                android.R.drawable.ic_lock_silent_mode
            }
        )
    }

    private fun speakRecognizedText(label: String) {
        if (!autoSpeakEnabled || label.isBlank()) return

        val textToSpeak = if (recognitionMode == RecognitionMode.STATIC) {
            getString(R.string.voice_letter_result, label)
        } else {
            getString(R.string.voice_sign_result, label)
        }
        val now = System.currentTimeMillis()
        if (textToSpeak == lastSpokenText && now - lastSpokenAt < AUTO_SPEAK_COOLDOWN_MS) {
            return
        }

        lastSpokenText = textToSpeak
        lastSpokenAt = now
        voiceOutputManager.speak(textToSpeak)
    }

    private fun showPermissionDeniedDialog() {
        AlertDialog.Builder(this)
            .setTitle(R.string.camera_permission_required_title)
            .setMessage(R.string.camera_permission_required_message)
            .setPositiveButton(R.string.accept_button_label, null)
            .show()
    }

    companion object {
        private const val STATIC_ANALYSIS_INTERVAL_MS = 300L
        private const val DYNAMIC_ANALYSIS_INTERVAL_MS = 120L
        private const val MIN_RECOGNITION_CONFIDENCE = 0.50f
        private const val MIN_RECOGNITION_MARGIN = 0.04f
        private const val STATIC_STABILITY_WINDOW = 5
        private const val MIN_STATIC_STABLE_VOTES = 2
        private const val MIN_STATIC_STABLE_RATIO = 0.40f
        private const val HIGH_CONFIDENCE_STATIC_THRESHOLD = 0.90f
        private const val HIGH_CONFIDENCE_STATIC_MARGIN = 0.12f
        private const val MIN_DYNAMIC_RECOGNITION_CONFIDENCE = 0.65f
        private const val MIN_DYNAMIC_RECOGNITION_MARGIN = 0.18f
        private const val DYNAMIC_STABILITY_WINDOW = 4
        private const val MIN_DYNAMIC_STABLE_VOTES = 2
        private const val MIN_DYNAMIC_STABLE_RATIO = 0.50f
        private const val DYNAMIC_RESULT_HOLD_MS = 4500L
        private const val AUTO_SPEAK_COOLDOWN_MS = 2500L
        private const val PREFERENCES_NAME = "signvoice_preferences"
        private const val PREF_TERMS_ACCEPTED = "terms_accepted"
    }

    private data class StaticPredictionVote(
        val label: String,
        val confidence: Float
    )

    private data class DynamicPredictionVote(
        val label: String,
        val confidence: Float
    )

    private enum class RecognitionMode {
        STATIC,
        DYNAMIC
    }

    private enum class DetectionVisualState {
        WAITING,
        RECOGNIZING,
        DETECTED,
        RETRY
    }

    private enum class AppSection {
        HOME,
        TRANSLATE,
        LEARN,
        HISTORY,
        PROFILE
    }
}
