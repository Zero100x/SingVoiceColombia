package com.singvoicecolombia.datasetcapture.utils

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

class VoiceOutputManager(context: Context) : TextToSpeech.OnInitListener {

    private var textToSpeech: TextToSpeech = TextToSpeech(context.applicationContext, this)
    private var isReady = false

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech.language = Locale("es", "CO")
            isReady = true
        }
    }

    fun speak(text: String) {
        if (!isReady || text.isBlank()) return
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "translation_result")
    }

    fun stop() {
        if (!isReady) return
        textToSpeech.stop()
    }

    fun shutdown() {
        textToSpeech.stop()
        textToSpeech.shutdown()
    }
}
