package com.example.my_application_week7

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide

class HeroesAdapter(private val heroesList: List<Pokemon>) :
    RecyclerView.Adapter<HeroesAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val heroesImage: ImageView = view.findViewById(R.id.heroes_image)
        val heroName: TextView = view.findViewById(R.id.hero_name)
        val heroAbility: TextView = view.findViewById(R.id.hero_ability)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_hero, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val hero = heroesList[position]
        holder.heroName.text = hero.name.capitalize()
        holder.heroAbility.text = "Ability: ${hero.ability.capitalize()}"

        Glide.with(holder.itemView)
            .load(hero.imageUrl)
            .into(holder.heroesImage)

        holder.itemView.setOnClickListener {
            Toast.makeText(
                holder.itemView.context,
                "${hero.name.capitalize()} clicked!",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    override fun getItemCount(): Int = heroesList.size
}
