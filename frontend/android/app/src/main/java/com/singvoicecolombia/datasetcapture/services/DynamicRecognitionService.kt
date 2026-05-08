package com.singvoicecolombia.datasetcapture.services

import android.content.Context
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.exp

data class DynamicRecognitionResult(
    val label: String,
    val confidence: Float,
    val ready: Boolean,
    val collectedFrames: Int,
    val requiredFrames: Int
)

class DynamicRecognitionService(private val context: Context) {
    private val labels: List<String> = loadLabels()
    private var interpreter: Interpreter? = loadInterpreterOrNull()
    private val frameBuffer = ArrayDeque<Array<Array<FloatArray>>>()

    fun isModelReady(): Boolean = interpreter != null && labels.isNotEmpty()

    fun reset() {
        frameBuffer.clear()
    }

    fun classify(imageProxy: ImageProxy): DynamicRecognitionResult? {
        val currentInterpreter = interpreter
        if (currentInterpreter == null || labels.isEmpty()) {
            imageProxy.close()
            return null
        }

        return try {
            val inputShape = currentInterpreter.getInputTensor(0).shape()
            val outputShape = currentInterpreter.getOutputTensor(0).shape()
            val inputHeight = inputShape.getOrNull(1) ?: 96
            val inputWidth = inputShape.getOrNull(2) ?: 96
            val inputChannels = inputShape.getOrNull(3) ?: 32
            val requiredFrames = (inputChannels / CHANNELS_PER_FRAME).coerceAtLeast(1)
            val outputSize = outputShape.lastOrNull() ?: labels.size

            val frameFeatures = buildFrameFeaturesFromLuma(imageProxy, inputHeight, inputWidth)
            frameBuffer.addLast(frameFeatures)

            while (frameBuffer.size > requiredFrames) {
                frameBuffer.removeFirst()
            }

            if (frameBuffer.size < requiredFrames) {
                DynamicRecognitionResult(
                    label = "",
                    confidence = 0f,
                    ready = false,
                    collectedFrames = frameBuffer.size,
                    requiredFrames = requiredFrames
                )
            } else {
                val input = buildInputFromBuffer(inputHeight, inputWidth, inputChannels)
                val output = Array(1) { FloatArray(outputSize) }
                currentInterpreter.run(input, output)

                val probabilities = normalizeOutput(output[0])
                val bestIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
                val label = labels.getOrElse(bestIndex) { "?" }

                DynamicRecognitionResult(
                    label = label,
                    confidence = probabilities[bestIndex].coerceIn(0f, 1f),
                    ready = true,
                    collectedFrames = frameBuffer.size,
                    requiredFrames = requiredFrames
                )
            }
        } catch (_: Exception) {
            null
        } finally {
            imageProxy.close()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        frameBuffer.clear()
    }

    private fun loadInterpreterOrNull(): Interpreter? {
        return try {
            Interpreter(loadModelFile(MODEL_FILE_NAME))
        } catch (_: Exception) {
            null
        }
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val modelBytes = context.assets.open(fileName).use { input ->
            input.readBytes()
        }
        return ByteBuffer.allocateDirect(modelBytes.size)
            .order(ByteOrder.nativeOrder())
            .apply {
                put(modelBytes)
                rewind()
            }
    }

    private fun loadLabels(): List<String> {
        return try {
            context.assets.open(LABELS_FILE_NAME).bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotBlank() }
                    .toList()
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun buildFrameFeaturesFromLuma(
        imageProxy: ImageProxy,
        targetHeight: Int,
        targetWidth: Int
    ): Array<Array<FloatArray>> {
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val imageWidth = imageProxy.width
        val imageHeight = imageProxy.height
        val lumaGrid = Array(targetHeight) { FloatArray(targetWidth) }
        val frameFeatures = Array(targetHeight) {
            Array(targetWidth) {
                FloatArray(CHANNELS_PER_FRAME)
            }
        }

        val roiSize = (minOf(imageWidth, imageHeight) * ROI_RATIO).toInt()
        val roiLeft = (imageWidth - roiSize) / 2
        val roiTop = (imageHeight - roiSize) / 2

        for (y in 0 until targetHeight) {
            val sourceY = roiTop + y * roiSize / targetHeight
            for (x in 0 until targetWidth) {
                val sourceX = roiLeft + x * roiSize / targetWidth
                val index = sourceY * rowStride + sourceX * pixelStride
                lumaGrid[y][x] = (buffer.get(index).toInt() and 0xFF).toFloat()
            }
        }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                frameFeatures[y][x][0] = lumaGrid[y][x]
                frameFeatures[y][x][1] = edgeMagnitude(lumaGrid, x, y)
            }
        }

        return frameFeatures
    }

    private fun buildInputFromBuffer(
        targetHeight: Int,
        targetWidth: Int,
        inputChannels: Int
    ): Array<Array<Array<FloatArray>>> {
        val input = Array(1) {
            Array(targetHeight) {
                Array(targetWidth) {
                    FloatArray(inputChannels)
                }
            }
        }

        frameBuffer.forEachIndexed { frameIndex, frameFeatures ->
            val channelOffset = frameIndex * CHANNELS_PER_FRAME
            for (y in 0 until targetHeight) {
                for (x in 0 until targetWidth) {
                    if (channelOffset < inputChannels) {
                        input[0][y][x][channelOffset] = frameFeatures[y][x][0]
                    }
                    if (channelOffset + 1 < inputChannels) {
                        input[0][y][x][channelOffset + 1] = frameFeatures[y][x][1]
                    }
                }
            }
        }

        return input
    }

    private fun edgeMagnitude(lumaGrid: Array<FloatArray>, x: Int, y: Int): Float {
        val maxY = lumaGrid.lastIndex
        val maxX = lumaGrid.firstOrNull()?.lastIndex ?: 0
        val left = lumaGrid[y][(x - 1).coerceAtLeast(0)]
        val right = lumaGrid[y][(x + 1).coerceAtMost(maxX)]
        val top = lumaGrid[(y - 1).coerceAtLeast(0)][x]
        val bottom = lumaGrid[(y + 1).coerceAtMost(maxY)][x]
        return ((abs(right - left) + abs(bottom - top)) * 0.5f).coerceIn(0f, 255f)
    }

    private fun normalizeOutput(rawValues: FloatArray): FloatArray {
        val alreadyProbabilities = rawValues.all { it in 0f..1f } && rawValues.sum() in 0.95f..1.05f
        if (alreadyProbabilities) return rawValues

        val maxValue = rawValues.maxOrNull() ?: 0f
        val exps = rawValues.map { exp((it - maxValue).toDouble()).toFloat() }
        val sum = exps.sum().takeIf { it > 0f } ?: 1f
        return exps.map { it / sum }.toFloatArray()
    }

    companion object {
        private const val MODEL_FILE_NAME = "modelo_dinamico.tflite"
        private const val LABELS_FILE_NAME = "dynamic_labels.txt"
        private const val CHANNELS_PER_FRAME = 2
        private const val ROI_RATIO = 0.75f
    }
}
