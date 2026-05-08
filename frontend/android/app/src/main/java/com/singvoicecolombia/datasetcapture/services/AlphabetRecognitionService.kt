package com.singvoicecolombia.datasetcapture.services

import android.content.Context
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

data class AlphabetRecognitionResult(
    val letter: String,
    val confidence: Float,
    val margin: Float,
    val signPresent: Boolean
)

class AlphabetRecognitionService(private val context: Context) {
    private val labels: List<String> = loadLabels()
    private var interpreter: Interpreter? = loadInterpreterOrNull()

    fun isModelReady(): Boolean = interpreter != null && labels.isNotEmpty()

    fun classify(imageProxy: ImageProxy): AlphabetRecognitionResult? {
        val currentInterpreter = interpreter
        if (currentInterpreter == null || labels.isEmpty()) {
            imageProxy.close()
            return null
        }

        return try {
            val inputShape = currentInterpreter.getInputTensor(0).shape()
            val outputShape = currentInterpreter.getOutputTensor(0).shape()
            val inputHeight = inputShape.getOrNull(1) ?: 224
            val inputWidth = inputShape.getOrNull(2) ?: 224
            val inputChannels = inputShape.getOrNull(3) ?: 3
            val outputSize = outputShape.lastOrNull() ?: labels.size

            val modelInput = buildInputFromImage(imageProxy, inputHeight, inputWidth, inputChannels)
            if (!modelInput.presenceStats.hasLikelyHandRegion()) {
                return AlphabetRecognitionResult(
                    letter = "",
                    confidence = 0f,
                    margin = 0f,
                    signPresent = false
                )
            }

            val output = Array(1) { FloatArray(outputSize) }
            currentInterpreter.run(modelInput.values, output)

            val probabilities = normalizeOutput(output[0])
            val rankedIndexes = probabilities.indices.sortedByDescending { probabilities[it] }
            val bestIndex = rankedIndexes.firstOrNull() ?: 0
            val secondIndex = rankedIndexes.getOrNull(1)
            val label = labels.getOrElse(bestIndex) { "?" }
            val confidence = probabilities[bestIndex].coerceIn(0f, 1f)
            val secondConfidence = secondIndex?.let { probabilities[it].coerceIn(0f, 1f) } ?: 0f

            AlphabetRecognitionResult(
                letter = label,
                confidence = confidence,
                margin = (confidence - secondConfidence).coerceAtLeast(0f),
                signPresent = true
            )
        } catch (_: Exception) {
            null
        } finally {
            imageProxy.close()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
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

    private fun buildInputFromImage(
        imageProxy: ImageProxy,
        targetHeight: Int,
        targetWidth: Int,
        channels: Int
    ): ModelInput {
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes.getOrNull(1)
        val vPlane = imageProxy.planes.getOrNull(2)
        val imageWidth = imageProxy.width
        val imageHeight = imageProxy.height
        val safeChannels = channels.coerceAtLeast(1)
        val input = Array(1) {
            Array(targetHeight) {
                Array(targetWidth) {
                    FloatArray(safeChannels)
                }
            }
        }
        val lumaGrid = Array(targetHeight) { FloatArray(targetWidth) }
        var skinLikePixels = 0
        var lumaSum = 0.0
        var lumaSquaredSum = 0.0

        val roiSize = (minOf(imageWidth, imageHeight) * ROI_RATIO).toInt()
        val roiLeft = (imageWidth - roiSize) / 2
        val roiTop = (imageHeight - roiSize) / 2

        for (y in 0 until targetHeight) {
            val sourceY = roiTop + y * roiSize / targetHeight
            for (x in 0 until targetWidth) {
                val sourceX = roiLeft + x * roiSize / targetWidth
                val luma = readPlaneValue(yPlane, sourceX, sourceY).toFloat()
                lumaGrid[y][x] = luma
                lumaSum += luma.toDouble()
                lumaSquaredSum += luma.toDouble() * luma.toDouble()

                if (uPlane != null && vPlane != null && isSkinLikePixel(luma, uPlane, vPlane, sourceX, sourceY)) {
                    skinLikePixels += 1
                }
            }
        }

        var edgeSum = 0f
        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val luma = lumaGrid[y][x]
                val edge = sobelEdgeMagnitude(lumaGrid, x, y)
                edgeSum += edge
                input[0][y][x][0] = luma

                if (safeChannels > 1) {
                    input[0][y][x][1] = edge
                }

                for (channel in 2 until safeChannels) {
                    input[0][y][x][channel] = luma
                }
            }
        }

        val sampleCount = (targetHeight * targetWidth).coerceAtLeast(1)
        val lumaMean = lumaSum / sampleCount
        val lumaVariance = (lumaSquaredSum / sampleCount) - (lumaMean * lumaMean)

        return ModelInput(
            values = input,
            presenceStats = PresenceStats(
                skinRatio = skinLikePixels.toFloat() / sampleCount.toFloat(),
                edgeMean = edgeSum / sampleCount.toFloat(),
                lumaStdDev = sqrt(max(lumaVariance, 0.0)).toFloat()
            )
        )
    }

    private fun readPlaneValue(plane: ImageProxy.PlaneProxy, x: Int, y: Int): Int {
        val index = y * plane.rowStride + x * plane.pixelStride
        val safeIndex = index.coerceIn(0, plane.buffer.limit() - 1)
        return plane.buffer.get(safeIndex).toInt() and 0xFF
    }

    private fun isSkinLikePixel(
        yValue: Float,
        uPlane: ImageProxy.PlaneProxy,
        vPlane: ImageProxy.PlaneProxy,
        sourceX: Int,
        sourceY: Int
    ): Boolean {
        val chromaX = sourceX / 2
        val chromaY = sourceY / 2
        val uValue = readPlaneValue(uPlane, chromaX, chromaY).toFloat()
        val vValue = readPlaneValue(vPlane, chromaX, chromaY).toFloat()

        val r = (yValue + 1.402f * (vValue - 128f)).coerceIn(0f, 255f)
        val g = (yValue - 0.344136f * (uValue - 128f) - 0.714136f * (vValue - 128f)).coerceIn(0f, 255f)
        val b = (yValue + 1.772f * (uValue - 128f)).coerceIn(0f, 255f)

        val skinByYuv = yValue >= MIN_SKIN_LUMA &&
            uValue in SKIN_CB_MIN..SKIN_CB_MAX &&
            vValue in SKIN_CR_MIN..SKIN_CR_MAX &&
            vValue >= uValue - 12f
        val skinByRgb = r > 35f &&
            g > 20f &&
            b > 10f &&
            maxOf(r, g, b) - minOf(r, g, b) > 8f &&
            r >= g - 12f &&
            r > b

        return skinByYuv || skinByRgb
    }

    private fun sobelEdgeMagnitude(lumaGrid: Array<FloatArray>, x: Int, y: Int): Float {
        val maxY = lumaGrid.lastIndex
        val maxX = lumaGrid.firstOrNull()?.lastIndex ?: 0
        val x0 = (x - 1).coerceAtLeast(0)
        val x1 = x
        val x2 = (x + 1).coerceAtMost(maxX)
        val y0 = (y - 1).coerceAtLeast(0)
        val y1 = y
        val y2 = (y + 1).coerceAtMost(maxY)

        val gx =
            -lumaGrid[y0][x0] + lumaGrid[y0][x2] +
            -2f * lumaGrid[y1][x0] + 2f * lumaGrid[y1][x2] +
            -lumaGrid[y2][x0] + lumaGrid[y2][x2]
        val gy =
            -lumaGrid[y0][x0] - 2f * lumaGrid[y0][x1] - lumaGrid[y0][x2] +
            lumaGrid[y2][x0] + 2f * lumaGrid[y2][x1] + lumaGrid[y2][x2]

        return ((abs(gx) + abs(gy)) * 0.5f).coerceIn(0f, 255f)
    }

    private fun normalizeOutput(rawValues: FloatArray): FloatArray {
        val alreadyProbabilities = rawValues.all { it in 0f..1f } && rawValues.sum() in 0.95f..1.05f
        if (alreadyProbabilities) return rawValues

        val maxValue = rawValues.maxOrNull() ?: 0f
        val exps = rawValues.map { exp((it - maxValue).toDouble()).toFloat() }
        val sum = exps.sum().takeIf { it > 0f } ?: 1f
        return exps.map { it / sum }.toFloatArray()
    }

    private data class ModelInput(
        val values: Array<Array<Array<FloatArray>>>,
        val presenceStats: PresenceStats
    )

    private data class PresenceStats(
        val skinRatio: Float,
        val edgeMean: Float,
        val lumaStdDev: Float
    ) {
        fun hasLikelyHandRegion(): Boolean {
            val hasSkin = skinRatio >= MIN_SKIN_RATIO
            val hasEdges = edgeMean >= MIN_EDGE_MEAN
            val hasContrast = lumaStdDev >= MIN_LUMA_STD_DEV
            return (hasSkin && hasEdges) ||
                (hasSkin && hasContrast) ||
                (hasEdges && hasContrast && edgeMean >= STRONG_EDGE_MEAN)
        }
    }

    companion object {
        private const val MODEL_FILE_NAME = "modelo_alfabeto.tflite"
        private const val LABELS_FILE_NAME = "labels.txt"
        private const val ROI_RATIO = 0.75f
        private const val MIN_SKIN_RATIO = 0.015f
        private const val MIN_EDGE_MEAN = 2.5f
        private const val STRONG_EDGE_MEAN = 5.0f
        private const val MIN_LUMA_STD_DEV = 8.0f
        private const val MIN_SKIN_LUMA = 30f
        private const val SKIN_CB_MIN = 65f
        private const val SKIN_CB_MAX = 158f
        private const val SKIN_CR_MIN = 100f
        private const val SKIN_CR_MAX = 205f
    }
}
