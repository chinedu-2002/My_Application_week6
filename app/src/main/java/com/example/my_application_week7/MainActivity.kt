package com.example.my_application_week7

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.codepath.asynchttpclient.AsyncHttpClient
import com.codepath.asynchttpclient.callback.JsonHttpResponseHandler
import okhttp3.Headers

class MainActivity : AppCompatActivity() {

    private lateinit var rvHeroes: RecyclerView
    private val heroesList = mutableListOf<Pokemon>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        rvHeroes = findViewById(R.id.heroes_list)
        rvHeroes.layoutManager = LinearLayoutManager(this)

        fetchPokemons()
    }

    private fun fetchPokemons() {
        val client = AsyncHttpClient()
        val url = "https://pokeapi.co/api/v2/pokemon?limit=20"

        client.get(url, object : JsonHttpResponseHandler() {
            override fun onSuccess(statusCode: Int, headers: Headers, json: JsonHttpResponseHandler.JSON) {
                val results = json.jsonObject.getJSONArray("results")

                for (i in 0 until results.length()) {
                    val pokeUrl = results.getJSONObject(i).getString("url")

                    client.get(pokeUrl, object : JsonHttpResponseHandler() {
                        override fun onSuccess(statusCode: Int, headers: Headers, json: JsonHttpResponseHandler.JSON) {
                            try {
                                val name = json.jsonObject.getString("name")
                                val imageUrl = json.jsonObject
                                    .getJSONObject("sprites")
                                    .getString("front_default")
                                val ability = json.jsonObject
                                    .getJSONArray("abilities")
                                    .getJSONObject(0)
                                    .getJSONObject("ability")
                                    .getString("name")

                                heroesList.add(Pokemon(name, imageUrl, ability))
                                rvHeroes.adapter = HeroesAdapter(heroesList)

                            } catch (e: Exception) {
                                Log.e("PokemonParse", "Error parsing Pokémon data", e)
                            }
                        }

                        override fun onFailure(statusCode: Int, headers: Headers?, response: String, throwable: Throwable?) {
                            Log.e("FetchDetails", "Failed fetching details: $response")
                        }
                    })
                }
            }

            override fun onFailure(statusCode: Int, headers: Headers?, response: String, throwable: Throwable?) {
                Log.e("FetchPokemons", "Initial fetch failed: $response")
            }
        })
    }
}
