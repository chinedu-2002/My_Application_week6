package com.example.my_application_week6

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.codepath.asynchttpclient.AsyncHttpClient
import com.codepath.asynchttpclient.callback.JsonHttpResponseHandler
import okhttp3.Headers
import kotlin.random.Random
import com.bumptech.glide.Glide

class MainActivity : AppCompatActivity() {
    private lateinit var pokemonImageView: ImageView
    private lateinit var pokemonName: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        pokemonImageView = findViewById(R.id.pokemonImageView)
        pokemonName = findViewById(R.id.pokemonName)
        val button = findViewById<Button>(R.id.next)
        setupButton(button)
        fetchHeroesImage()
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }
    private fun fetchHeroesImage() {
        val client = AsyncHttpClient()
        var choice = Random.nextInt(60)
        val heroesImageURL = "https://pokeapi.co/api/v2/pokemon/$choice"
        Log.d("heroesImageURL", "heroes image URL set: $heroesImageURL")

        client[heroesImageURL, object : JsonHttpResponseHandler() {
            override fun onSuccess(statusCode: Int, headers: Headers, json: JsonHttpResponseHandler.JSON) {
                Log.d("Heroes", "response successful: $json")

                val imageUrl = json.jsonObject
                    .getJSONObject("sprites")
                    .getString("front_default")

                val name = json.jsonObject
                    .getString("name")
                pokemonName.text = name




                Glide.with(this@MainActivity)
                    .load(imageUrl)
                    .fitCenter()
                    .into(pokemonImageView)
            }



            override fun onFailure(
                statusCode: Int,
                headers: Headers?,
                errorResponse: String,
                throwable: Throwable?
            ) {
                Log.d("Heroes Error", errorResponse)
            }
        }]


    }

    private fun setupButton(button: Button) {
        button.setOnClickListener {
            fetchHeroesImage()
        }

    }


}